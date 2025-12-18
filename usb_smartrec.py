"""
smartrec.py

Small, self-contained Smart Recording manager for DeepStream Python.

Supports two modes:
1) nvurisrcbin built-in Smart Record (recommended for RTSP/file sources)
2) Manual NvDsSR (ctypes bindings) for sources that don't support built-in SR (e.g., USB/v4l2)

This module is designed to be imported and used from Forklift.py.
"""

from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

import pyds  # noqa: E402

# -----------------------------------------------------------------------------
# NvDsSR library loading
# -----------------------------------------------------------------------------

_LIB_CANDIDATES = (
    "/opt/nvidia/deepstream/deepstream/lib/libnvdsgst_smartrecord.so",  # common DeepStream 6/7
)

_libnvds_sr = None
for _path in _LIB_CANDIDATES:
    try:
        _libnvds_sr = ctypes.CDLL(_path)
        print(_libnvds_sr)
        break
    except OSError:
        _libnvds_sr = None

# -----------------------------------------------------------------------------
# ctypes structures / bindings
# -----------------------------------------------------------------------------

NvDsSRContainerType = ctypes.c_int
NVDSSR_CONTAINER_MP4 = 0
NVDSSR_CONTAINER_MKV = 1

SR_CALLBACK_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)


class NvDsSRInitParams(ctypes.Structure):
    _fields_ = [
        ("callback", SR_CALLBACK_FUNC),
        ("containerType", NvDsSRContainerType),
        ("width", ctypes.c_uint),
        ("height", ctypes.c_uint),
        ("fileNamePrefix", ctypes.c_char_p),
        ("dirpath", ctypes.c_char_p),
        ("defaultDuration", ctypes.c_uint),
        ("cacheSize", ctypes.c_uint),
    ]


class NvDsSRContext(ctypes.Structure):
    """
    Partial representation. We only need recordbin and a stable pointer for Start/Stop/Destroy.
    """
    _fields_ = [
        ("recordbin", ctypes.c_void_p),  # GstElement*
        ("recordQue", ctypes.c_void_p),
        ("encodebin", ctypes.c_void_p),
        ("filesink", ctypes.c_void_p),
        ("gotKeyFrame", ctypes.c_bool),
        ("recordOn", ctypes.c_bool),
        ("resetDone", ctypes.c_bool),
        ("isPlaying", ctypes.c_bool),
        ("initParams", NvDsSRInitParams),
    ]


class SRUserContext(ctypes.Structure):
    _fields_ = [
        ("sessionid", ctypes.c_int),
        ("name", ctypes.c_char * 32),
    ]


def _capsule_ptr(capsule) -> int:
    """
    pyds.get_native_ptr() returns a PyCapsule.
    We need the raw address for ctypes operations.
    """
    if capsule is None:
        return 0
    try:
        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        return int(ctypes.pythonapi.PyCapsule_GetPointer(capsule, None) or 0)
    except Exception:
        # Fallback: hash() often maps to the underlying pointer for GI objects, but not guaranteed.
        try:
            return int(hash(capsule))
        except Exception:
            return 0


if _libnvds_sr:
    _libnvds_sr.NvDsSRCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(NvDsSRContext)), ctypes.POINTER(NvDsSRInitParams)]
    _libnvds_sr.NvDsSRCreate.restype = ctypes.c_int

    _libnvds_sr.NvDsSRStart.argtypes = [
        ctypes.POINTER(NvDsSRContext),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.c_uint,  # startTime (seconds in the past)
        ctypes.c_uint,  # duration (seconds)
        ctypes.c_void_p,  # userData
    ]
    _libnvds_sr.NvDsSRStart.restype = ctypes.c_int

    _libnvds_sr.NvDsSRStop.argtypes = [ctypes.POINTER(NvDsSRContext), ctypes.c_uint]  # sessionId
    _libnvds_sr.NvDsSRStop.restype = ctypes.c_int

    _libnvds_sr.NvDsSRDestroy.argtypes = [ctypes.POINTER(NvDsSRContext)]
    _libnvds_sr.NvDsSRDestroy.restype = ctypes.c_int


@dataclass
class _NativeBuffers:
    """Holds gbuffer objects so they stay alive for the lifetime of the SR attachment."""
    session_gbuf: object
    uctx_gbuf: object
    params_gbuf: Optional[object] = None


class SmartRecManager:
    """
    Smart Record manager.

    - attach_source(sid, nvurisrcbin, is_manual=False): enables built-in Smart Record
    - attach_source(sid, tee_or_any, is_manual=True): creates a manual NvDsSR context and returns recordbin_ptr (int)

    start(sid, back_sec, front_sec): starts recording
    stop(sid): stops recording
    """

    def __init__(self, record_dir: str, cache_sec: int = 60, file_prefix: str = "cam"):
        self._record_dir = os.path.abspath(record_dir)
        self._cache_sec = int(cache_sec)
        self._prefix = str(file_prefix)

        os.makedirs(self._record_dir, exist_ok=True)

        # nvurisrcbin mode
        self._nvuri_by_sid: Dict[int, Gst.Element] = {}

        # manual mode: store POINTER(NvDsSRContext)
        self._manual_ctx_by_sid: Dict[int, ctypes.POINTER(NvDsSRContext)] = {}

        # Keep native buffers alive (session id + user ctx + init params)
        self._native: Dict[int, _NativeBuffers] = {}

        # Keep the PyCapsules (required by nvurisrcbin start-sr / stop-sr)
        self._session_capsule: Dict[int, object] = {}
        self._uctx_capsule: Dict[int, object] = {}

        # Raw pointers for manual mode (ctypes)
        self._uctx_raw_ptr: Dict[int, int] = {}

        # Optional anti-spam
        self._cooldown_s: float = 0.0
        self._last_fire: Dict[int, float] = {}

        # Keep C callback references alive (important!)
        self._c_callbacks: Dict[int, object] = {}

    def set_cooldown(self, seconds: float) -> None:
        self._cooldown_s = max(0.0, float(seconds))

    # ------------------------------------------------------------------
    # Wiring / attachment
    # ------------------------------------------------------------------

    def attach_source(self, sid: int, source_elem: Gst.Element, friendly_name: Optional[str] = None, is_manual: bool = False) -> Optional[int]:
        """
        Attach a source for Smart Record.

        If is_manual is False:
            - source_elem MUST be nvurisrcbin
            - returns None

        If is_manual is True:
            - creates NvDsSR context using ctypes
            - returns recordbin_ptr (int) so the caller can gst_bin_add + link it
        """
        if not isinstance(source_elem, Gst.Element):
            raise TypeError("attach_source expects a Gst.Element")

        # Allocate native buffers for session id and SRUserContext
        sess_g = pyds.alloc_buffer(4)
        sess_capsule = pyds.get_native_ptr(sess_g)

        uctx_size = ctypes.sizeof(SRUserContext)
        uctx_g = pyds.alloc_buffer(uctx_size)
        uctx_capsule = pyds.get_native_ptr(uctx_g)

        uctx_ptr = _capsule_ptr(uctx_capsule)
        if uctx_ptr == 0:
            raise RuntimeError("Failed to obtain raw pointer for SRUserContext")

        # Initialize SRUserContext contents
        sr = ctypes.cast(uctx_ptr, ctypes.POINTER(SRUserContext)).contents
        sr.sessionid = int(time.time()) & 0x7FFFFFFF
        name = (friendly_name or f"sid{sid}").encode("utf-8")[:31]
        sr.name = name.ljust(31, b"\0")

        self._native[sid] = _NativeBuffers(session_gbuf=sess_g, uctx_gbuf=uctx_g)
        self._session_capsule[sid] = sess_capsule
        self._uctx_capsule[sid] = uctx_capsule
        self._uctx_raw_ptr[sid] = uctx_ptr

        if not is_manual:
            # nvurisrcbin mode
            nv = source_elem
            nv.set_property("smart-record", 2)
            nv.set_property("smart-rec-dir-path", self._record_dir)
            try:
                nv.set_property("smart-rec-cache", self._cache_sec)
            except Exception:
                pass
            try:
                nv.set_property("smart-rec-file-prefix", f"{self._prefix}{sid}_")
            except Exception:
                pass

            self._nvuri_by_sid[sid] = nv

            # Optional: print file info when done
            try:
                nv.connect("sr-done", self._on_nvuri_done, sid)
            except Exception:
                pass

            return None

        # Manual mode
        if not _libnvds_sr:
            print("[SmartRec] WARNING: NvDsSR library not found; manual SR is disabled.")
            return None

        params_size = ctypes.sizeof(NvDsSRInitParams)
        params_g = pyds.alloc_buffer(params_size)
        params_capsule = pyds.get_native_ptr(params_g)
        params_ptr = _capsule_ptr(params_capsule)
        if params_ptr == 0:
            print("[SmartRec] ERROR: Failed to get InitParams raw pointer.")
            return None

        # Keep params buffer alive
        self._native[sid].params_gbuf = params_g

        # Create (and pin) callback
        cb = SR_CALLBACK_FUNC(self._on_manual_done_c)
        self._c_callbacks[sid] = cb

        init_params = ctypes.cast(params_ptr, ctypes.POINTER(NvDsSRInitParams)).contents
        init_params.callback = cb
        init_params.containerType = NVDSSR_CONTAINER_MP4
        init_params.width = 0
        init_params.height = 0
        init_params.fileNamePrefix = f"{self._prefix}{sid}_".encode("utf-8")
        init_params.dirpath = self._record_dir.encode("utf-8")
        init_params.defaultDuration = 10
        init_params.cacheSize = self._cache_sec

        ctx_ptr = ctypes.POINTER(NvDsSRContext)()
        ret = _libnvds_sr.NvDsSRCreate(ctypes.byref(ctx_ptr), ctypes.cast(params_ptr, ctypes.POINTER(NvDsSRInitParams)))
        if ret != 0 or not ctx_ptr:
            print(f"[SmartRec] ERROR: NvDsSRCreate failed (sid={sid}, ret={ret}).")
            return None

        self._manual_ctx_by_sid[sid] = ctx_ptr
        recordbin_ptr = int(ctx_ptr.contents.recordbin or 0)
        return recordbin_ptr or None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, sid: int, back_sec: int, front_sec: int, label: Optional[str] = None) -> bool:
        """
        Start Smart Record for the given source ID.

        - back_sec: seconds in the past (uses Smart Record circular cache)
        - front_sec: seconds after trigger
        """
        if self._cooldown_s > 0:
            last = self._last_fire.get(sid, 0.0)
            now = time.monotonic()
            if now - last < self._cooldown_s:
                return False

        # Update label (stored in SRUserContext)
        if label and sid in self._uctx_raw_ptr:
            try:
                uctx_ptr = self._uctx_raw_ptr[sid]
                sr = ctypes.cast(uctx_ptr, ctypes.POINTER(SRUserContext)).contents
                nm = label.encode("utf-8")[:31]
                sr.name = nm.ljust(31, b"\0")
            except Exception:
                pass

        # Manual mode
        if sid in self._manual_ctx_by_sid and _libnvds_sr:
            ctx = self._manual_ctx_by_sid[sid]
            session_id = ctypes.c_uint(0)
            total_dur = int(back_sec) + int(front_sec)
            user_data = ctypes.c_void_p(int(self._uctx_raw_ptr.get(sid, 0)) or 0)

            ret = _libnvds_sr.NvDsSRStart(ctx, ctypes.byref(session_id), int(back_sec), int(total_dur), user_data)
            if ret == 0:
                self._last_fire[sid] = time.monotonic()
                return True
            return False

        # nvurisrcbin mode
        nv = self._nvuri_by_sid.get(sid)
        if nv is None:
            return False

        # Respect cache size if readable
        try:
            cache = int(nv.get_property("smart-rec-cache") or 0)
            if cache > 0:
                back_sec = min(int(back_sec), cache)
        except Exception:
            pass

        sess_caps = self._session_capsule.get(sid)
        uctx_caps = self._uctx_capsule.get(sid)

        try:
            nv.emit("start-sr", sess_caps, int(back_sec), int(front_sec), uctx_caps)
            self._last_fire[sid] = time.monotonic()
            return True
        except TypeError:
            # Some builds require user_ctx=None
            try:
                nv.emit("start-sr", sess_caps, int(back_sec), int(front_sec), None)
                self._last_fire[sid] = time.monotonic()
                return True
            except Exception:
                return False
        except Exception:
            return False

    def stop(self, sid: int) -> bool:
        """Stop recording (best-effort)."""
        if sid in self._manual_ctx_by_sid and _libnvds_sr:
            ctx = self._manual_ctx_by_sid[sid]
            ret = _libnvds_sr.NvDsSRStop(ctx, 0)
            return ret == 0

        nv = self._nvuri_by_sid.get(sid)
        if nv is None:
            return False

        try:
            try:
                nv.emit("stop-sr", 0)
            except TypeError:
                nv.emit("stop-sr")
            return True
        except Exception:
            return False

    def link_recordbin(self, gst_bin: Gst.Bin, recordbin_ptr: int, source_elem: Gst.Element) -> bool:
        """
        Manual SR helper: add recordbin (GstElement*) into gst_bin, then link source_elem -> recordbin.

        Why ctypes?
        - recordbin_ptr is a C pointer that GI can't wrap into a Python Gst.Element easily.
        """
        if not recordbin_ptr:
            return False

        try:
            libgst = ctypes.CDLL("libgstreamer-1.0.so.0")
        except OSError:
            print("[SmartRec] ERROR: Could not load libgstreamer-1.0.so.0")
            return False

        libgst.gst_bin_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        libgst.gst_bin_add.restype = ctypes.c_bool

        libgst.gst_element_link.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        libgst.gst_element_link.restype = ctypes.c_bool

        libgst.gst_element_sync_state_with_parent.argtypes = [ctypes.c_void_p]
        libgst.gst_element_sync_state_with_parent.restype = ctypes.c_bool

        # In PyGObject, hash(obj) usually returns the underlying GObject pointer value.
        bin_ptr = ctypes.c_void_p(int(hash(gst_bin)))
        src_ptr = ctypes.c_void_p(int(hash(source_elem)))
        rec_ptr = ctypes.c_void_p(int(recordbin_ptr))

        if not libgst.gst_bin_add(bin_ptr, rec_ptr):
            return False
        if not libgst.gst_element_link(src_ptr, rec_ptr):
            return False

        libgst.gst_element_sync_state_with_parent(rec_ptr)
        return True

    def cleanup(self) -> None:
        """Destroy manual SR contexts (best-effort)."""
        if _libnvds_sr:
            for sid, ctx in list(self._manual_ctx_by_sid.items()):
                try:
                    _libnvds_sr.NvDsSRDestroy(ctx)
                except Exception:
                    pass
        self._manual_ctx_by_sid.clear()
        self._nvuri_by_sid.clear()
        self._native.clear()
        self._session_capsule.clear()
        self._uctx_capsule.clear()
        self._uctx_raw_ptr.clear()
        self._c_callbacks.clear()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_nvuri_done(self, _nvurisrcbin, recordingInfo, _user_ctx, sid: int) -> None:
        """nvurisrcbin sr-done callback (prints file info if available)."""
        try:
            info = pyds.NvDsSRRecordingInfo.cast(hash(recordingInfo))
            try:
                dirp = pyds.get_string(info.dirpath)
                filep = pyds.get_string(info.filename)
            except Exception:
                dirp = getattr(info, "dirpath", b"")
                filep = getattr(info, "filename", b"")
            print(f"[SR DONE] sid={sid} file={filep} dir={dirp}")
        except Exception as e:
            print("[SR DONE] error:", e)

    def _on_manual_done_c(self, info_p, _user_data_p) -> None:
        """Manual NvDsSR callback (best-effort)."""
        try:
            info = pyds.NvDsSRRecordingInfo.cast(info_p)
            try:
                dirp = pyds.get_string(info.dirpath)
                filep = pyds.get_string(info.filename)
            except Exception:
                dirp = b"?"
                filep = b"?"
            print(f"[SR DONE MANUAL] file={filep} dir={dirp}")
        except Exception as e:
            print("[SR DONE MANUAL] error:", e)
