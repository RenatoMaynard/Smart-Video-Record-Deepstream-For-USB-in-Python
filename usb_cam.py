#!/usr/bin/env python3
"""
usb_cam_demo_single.py

Single USB camera DeepStream demo with Smart Recording.

What this example shows:
- USB Smart Record using **manual NvDsSR** (because USB/v4l2 sources do not support nvurisrcbin Smart Record):
    * Auto-trigger when a **person** is detected (with confidence >= MIN_PERSON_CONF)
    * Manual trigger by pressing **R**

Overlay behavior:
- Only the person bounding box is shown.
- The label is:  "person 87.3%"  (confidence from the model).
- All other detected classes are hidden (no box, no text).

Requirements:
- NVIDIA Jetson / DeepStream Python environment (gi + pyds)
- smartrec_clean.py in the same folder (or in PYTHONPATH)
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
gi.require_version("Gtk", "3.0")

from gi.repository import Gst, GstVideo, Gtk, Gdk  # noqa: E402

import pyds  # noqa: E402

# IMPORTANT: include smartrec_clean.py in your repo next to this file
from usb_smartrec import SmartRecManager  # noqa: E402


# -----------------------------------------------------------------------------
# CONFIG (edit these for your machine)
# -----------------------------------------------------------------------------

USB_DEVICE = "/dev/video0"

# Request MJPEG from the camera (common for USB cameras on Jetson). Adjust if needed.
CAM_W, CAM_H, CAM_FPS = 1280, 720, 30
USB_CAPS = f"image/jpeg,width={CAM_W},height={CAM_H},framerate={CAM_FPS}/1"

# DeepStream primary detector config (must exist)
PGIE_CONFIG = "/home/nvidia/Desktop/new/dstest1_pgie_config.txt"

# Keep mux resolution the same as camera for a simple demo.
MUX_W, MUX_H = CAM_W, CAM_H

# Smart Record output
SMARTREC_DIR = "SmartRecDir"
SR_CACHE_SEC = 60          # circular cache size (seconds)
SR_BACK_SEC = 10           # seconds before trigger
SR_FRONT_SEC = 10          # seconds after trigger
SR_COOLDOWN_SEC = 60.0     # seconds between auto triggers

# Person identification
PERSON_CLASS_IDS = {0}     # often 0; update if your model differs
MIN_PERSON_CONF = 0.35     # show/trigger only if confidence >= this

# DeepStream env tweak (safe)
os.environ["NVSTREAMMUX_ADAPTIVE_BATCHING"] = "yes"


# -----------------------------------------------------------------------------
# Smart Record manager (manual mode for USB)
# -----------------------------------------------------------------------------

SRM = SmartRecManager(SMARTREC_DIR, cache_sec=SR_CACHE_SEC, file_prefix="cam")
SRM.set_cooldown(SR_COOLDOWN_SEC)
_last_auto_trigger = 0.0


# -----------------------------------------------------------------------------
# GStreamer helpers
# -----------------------------------------------------------------------------

def _request_tee_src_pad(tee: Gst.Element) -> Gst.Pad:
    """Request a new tee src pad without deprecated get_request_pad()."""
    tpl = tee.get_pad_template("src_%u")
    if not tpl:
        raise RuntimeError("tee has no 'src_%u' pad template")
    pad = tee.request_pad(tpl, None, None)
    if not pad:
        raise RuntimeError("Failed to request tee src pad")
    return pad


def _make_queue(name: str, leaky: bool, max_buf: int) -> Gst.Element:
    """Create a queue with small buffering (good for low-latency display)."""
    q = Gst.ElementFactory.make("queue", name)
    if not q:
        raise RuntimeError(f"Failed to create queue '{name}'")

    if leaky:
        q.set_property("leaky", 2)  # downstream
    q.set_property("max-size-buffers", int(max_buf))
    q.set_property("max-size-time", 0)
    q.set_property("max-size-bytes", 0)
    return q


# -----------------------------------------------------------------------------
# OSD probe: hide non-person classes + show person confidence + SR auto trigger
# -----------------------------------------------------------------------------

def _get_obj_label(obj: pyds.NvDsObjectMeta) -> str:
    """Best-effort label (some models set obj_label; others rely on class_id)."""
    try:
        return pyds.get_string(obj.obj_label).strip().lower()
    except Exception:
        return ""


def _get_obj_conf(obj: pyds.NvDsObjectMeta) -> float:
    """Best-effort confidence."""
    try:
        return float(getattr(obj, "confidence", -1.0))
    except Exception:
        return -1.0


def osd_sink_probe(_pad: Gst.Pad, info: Gst.PadProbeInfo, _u_data) -> Gst.PadProbeReturn:
    """
    Runs on each frame at nvdsosd sink:
    - Keep only person boxes visible
    - Set label text to: "person XX.X%"
    - Auto-trigger Smart Record when a person is present (with cooldown)
    """
    global _last_auto_trigger

    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    saw_person = False

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            try:
                class_id = int(getattr(obj, "class_id", -1))
                label = _get_obj_label(obj)
                conf = _get_obj_conf(obj)

                is_person = (class_id in PERSON_CLASS_IDS) or (label == "person")
                show = bool(is_person and conf >= MIN_PERSON_CONF)

                # Hide everything by default
                if not show:
                    obj.rect_params.border_width = 0
                    obj.text_params.display_text = ""
                else:
                    saw_person = True
                    obj.rect_params.border_width = 3
                    # Keep default OSD border color; you can set it if you want.
                    obj.text_params.display_text = f"person {conf * 100.0:.1f}%"
            except Exception:
                # Never crash the pipeline on overlay issues
                pass

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    # Auto-trigger Smart Record when a person is visible
    if saw_person:
        now = time.monotonic()
        if (now - _last_auto_trigger) >= SR_COOLDOWN_SEC:
            ok = SRM.start(sid=0, back_sec=SR_BACK_SEC, front_sec=SR_FRONT_SEC, label="person")
            if ok:
                print("[SR] Auto-trigger started (person)")
            _last_auto_trigger = now

    return Gst.PadProbeReturn.OK


# -----------------------------------------------------------------------------
# GTK fullscreen window (VideoOverlay)
# -----------------------------------------------------------------------------

class FullscreenWindow:
    """Minimal fullscreen GTK window for nveglglessink via GstVideoOverlay."""

    def __init__(self, sink: Gst.Element):
        self.sink = sink

        self.win = Gtk.Window()
        self.win.set_title("USB Camera Demo (DeepStream)")
        self.win.fullscreen()

        self.area = Gtk.DrawingArea()
        self.area.set_hexpand(True)
        self.area.set_vexpand(True)
        self.win.add(self.area)

        self.win.connect("destroy", self._on_destroy)
        self.win.connect("key-press-event", self._on_key)
        self.area.connect("realize", self._on_realize)

        self.win.show_all()

    def _on_realize(self, widget) -> None:
        gdk_window = widget.get_window()
        if not gdk_window:
            return
        xid = gdk_window.get_xid()
        try:
            GstVideo.VideoOverlay.set_window_handle(self.sink, xid)
        except Exception:
            # Fallback property name in some builds
            try:
                self.sink.set_property("window-xid", xid)
            except Exception:
                pass

    def _on_key(self, _w, event) -> bool:
        if event.keyval == Gdk.KEY_Escape:
            Gtk.main_quit()
            return True

        if event.keyval in (Gdk.KEY_r, Gdk.KEY_R):
            ok = SRM.start(sid=0, back_sec=SR_BACK_SEC, front_sec=SR_FRONT_SEC, label="key_R")
            if ok:
                print("[SR] Manual trigger started (key_R)")
            return True

        return False

    def _on_destroy(self, *_args) -> None:
        Gtk.main_quit()


# -----------------------------------------------------------------------------
# Pipeline construction (USB only)
# -----------------------------------------------------------------------------

@dataclass
class PipelineParts:
    pipeline: Gst.Pipeline
    sink: Gst.Element


def build_usb_pipeline() -> PipelineParts:
    """
    Build a USB-only pipeline with a tee:
    - Branch A: inference + display
    - Branch B: encode + manual Smart Record recordbin
    """
    pipe = Gst.Pipeline.new("usb_singlecam_demo")
    if not pipe:
        raise RuntimeError("Failed to create pipeline")

    # USB source -> MJPEG -> NVMM (NV12)
    v4l2 = Gst.ElementFactory.make("v4l2src", "v4l2src")
    caps_mjpg = Gst.ElementFactory.make("capsfilter", "caps_mjpg")
    dec = Gst.ElementFactory.make("nvv4l2decoder", "mjpeg_dec")
    nvvidconv = Gst.ElementFactory.make("nvvidconv", "nvvidconv")
    caps_nvmm = Gst.ElementFactory.make("capsfilter", "caps_nvmm")
    tee = Gst.ElementFactory.make("tee", "tee")

    if not all([v4l2, caps_mjpg, dec, nvvidconv, caps_nvmm, tee]):
        raise RuntimeError("Failed to create USB decode elements")

    v4l2.set_property("device", USB_DEVICE)
    caps_mjpg.set_property("caps", Gst.Caps.from_string(USB_CAPS))

    # Many MJPEG USB cams require this on Jetson
    try:
        dec.set_property("mjpeg", 1)
    except Exception:
        pass

    caps_nvmm.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12"))
    tee.set_property("allow-not-linked", True)

    # ---------------- Branch A: inference + display ----------------
    q_main = _make_queue("q_main", leaky=True, max_buf=1)

    mux = Gst.ElementFactory.make("nvstreammux", "mux")
    caps_post_mux = Gst.ElementFactory.make("capsfilter", "caps_post_mux")
    pgie = Gst.ElementFactory.make("nvinfer", "pgie")
    conv_rgba = Gst.ElementFactory.make("nvvideoconvert", "conv_rgba")
    caps_rgba = Gst.ElementFactory.make("capsfilter", "caps_rgba")
    osd = Gst.ElementFactory.make("nvdsosd", "osd")
    egltrans = Gst.ElementFactory.make("nvegltransform", "egltrans")
    sink = Gst.ElementFactory.make("nveglglessink", "sink")

    if not all([mux, caps_post_mux, pgie, conv_rgba, caps_rgba, osd, egltrans, sink]):
        raise RuntimeError("Failed to create display/infer elements")

    mux.set_property("batch-size", 1)
    mux.set_property("live-source", 1)
    mux.set_property("width", int(MUX_W))
    mux.set_property("height", int(MUX_H))
    mux.set_property("batched-push-timeout", int(1_000_000 / max(1, CAM_FPS)))
    try:
        mux.set_property("sync-inputs", 0)
    except Exception:
        pass

    caps_post_mux.set_property(
        "caps",
        Gst.Caps.from_string(f"video/x-raw(memory:NVMM),width={MUX_W},height={MUX_H},framerate={CAM_FPS}/1"),
    )

    # nvinfer: unique-id MUST be set (fixes 'Unique ID not set')
    pgie.set_property("config-file-path", PGIE_CONFIG)
    pgie.set_property("unique-id", 1)
    try:
        pgie.set_property("batch-size", 1)
    except Exception:
        pass

    caps_rgba.set_property(
        "caps",
        Gst.Caps.from_string(f"video/x-raw(memory:NVMM),format=RGBA,width={MUX_W},height={MUX_H},framerate={CAM_FPS}/1"),
    )

    sink.set_property("sync", 0)
    sink.set_property("qos", 1)
    try:
        sink.set_property("force-aspect-ratio", True)
    except Exception:
        pass

    # ---------------- Branch B: encode + Smart Record ----------------
    q_sr = _make_queue("q_sr", leaky=False, max_buf=30)
    sr_conv = Gst.ElementFactory.make("nvvideoconvert", "sr_conv")

    # Prefer hardware encoder on Jetson; fallback to software encoder.
    sr_enc = Gst.ElementFactory.make("nvv4l2h264enc", "sr_enc") or Gst.ElementFactory.make("openh264enc", "sr_enc_sw")
    sr_parse = Gst.ElementFactory.make("h264parse", "sr_parse")

    if not all([q_sr, sr_conv, sr_enc, sr_parse]):
        raise RuntimeError("Failed to create SR encode elements")

    # Encoder tuning (best-effort)
    for prop, val in (("bitrate", 8_000_000), ("insert-sps-pps", 1), ("iframeinterval", 30)):
        try:
            sr_enc.set_property(prop, val)
        except Exception:
            pass

    # Add all elements
    for e in (
        v4l2, caps_mjpg, dec, nvvidconv, caps_nvmm, tee,
        q_main, mux, caps_post_mux, pgie, conv_rgba, caps_rgba, osd, egltrans, sink,
        q_sr, sr_conv, sr_enc, sr_parse,
    ):
        pipe.add(e)

    # Link decode -> tee
    if not v4l2.link(caps_mjpg):
        raise RuntimeError("Link failed: v4l2src -> caps_mjpg")
    if not caps_mjpg.link(dec):
        raise RuntimeError("Link failed: caps_mjpg -> decoder")
    if not dec.link(nvvidconv):
        raise RuntimeError("Link failed: decoder -> nvvidconv")
    if not nvvidconv.link(caps_nvmm):
        raise RuntimeError("Link failed: nvvidconv -> caps_nvmm")
    if not caps_nvmm.link(tee):
        raise RuntimeError("Link failed: caps_nvmm -> tee")

    # Branch A: tee -> q_main -> mux.sink_0 -> caps_post_mux -> pgie -> conv_rgba -> caps_rgba -> osd -> egltrans -> sink
    tee_src_main = _request_tee_src_pad(tee)
    if tee_src_main.link(q_main.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
        raise RuntimeError("Link failed: tee -> q_main")

    mux_sink0 = mux.request_pad_simple("sink_0")
    if not mux_sink0:
        raise RuntimeError("Could not request mux.sink_0")
    if q_main.get_static_pad("src").link(mux_sink0) != Gst.PadLinkReturn.OK:
        raise RuntimeError("Link failed: q_main.src -> mux.sink_0")

    if not mux.link(caps_post_mux):
        raise RuntimeError("Link failed: mux -> caps_post_mux")
    if not caps_post_mux.link(pgie):
        raise RuntimeError("Link failed: caps_post_mux -> pgie")
    if not pgie.link(conv_rgba):
        raise RuntimeError("Link failed: pgie -> conv_rgba")
    if not conv_rgba.link(caps_rgba):
        raise RuntimeError("Link failed: conv_rgba -> caps_rgba")
    if not caps_rgba.link(osd):
        raise RuntimeError("Link failed: caps_rgba -> osd")
    if not osd.link(egltrans):
        raise RuntimeError("Link failed: osd -> egltrans")
    if not egltrans.link(sink):
        raise RuntimeError("Link failed: egltrans -> sink")

    # OSD probe: hide non-person + show confidence + auto SR
    osd_sink_pad = osd.get_static_pad("sink")
    if not osd_sink_pad:
        raise RuntimeError("Failed to get osd sink pad")
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_probe, None)

    # Branch B: tee -> q_sr -> sr_conv -> sr_enc -> sr_parse -> recordbin
    tee_src_sr = _request_tee_src_pad(tee)
    if tee_src_sr.link(q_sr.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
        raise RuntimeError("Link failed: tee -> q_sr")
    if not q_sr.link(sr_conv):
        raise RuntimeError("Link failed: q_sr -> sr_conv")
    if not sr_conv.link(sr_enc):
        raise RuntimeError("Link failed: sr_conv -> sr_enc")
    if not sr_enc.link(sr_parse):
        raise RuntimeError("Link failed: sr_enc -> sr_parse")

    # Attach manual Smart Record to this source (sid=0)
    recordbin_ptr = SRM.attach_source(0, tee, friendly_name="cam0", is_manual=True)
    if recordbin_ptr:
        ok = SRM.link_recordbin(pipe, recordbin_ptr, sr_parse)
        if ok:
            print("[SR] Manual NvDsSR recordbin linked (USB).")
        else:
            print("[SR] WARNING: Failed to link recordbin (recording disabled).")
    else:
        print("[SR] WARNING: Manual SR not available (library missing or attach failed).")

    return PipelineParts(pipeline=pipe, sink=sink)


# -----------------------------------------------------------------------------
# Bus handling
# -----------------------------------------------------------------------------

def on_bus_message(_bus: Gst.Bus, msg: Gst.Message, _user_data) -> None:
    t = msg.type
    if t == Gst.MessageType.ERROR:
        err, dbg = msg.parse_error()
        sys.stderr.write(f"\n[GStreamer ERROR] {err}\n{dbg}\n")
        Gtk.main_quit()
    elif t == Gst.MessageType.EOS:
        print("[GStreamer] EOS")
        Gtk.main_quit()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    Gst.init(None)

    parts = build_usb_pipeline()
    pipeline = parts.pipeline
    sink = parts.sink

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus_message, None)

    pipeline.set_state(Gst.State.PLAYING)

    _ui = FullscreenWindow(sink)

    try:
        Gtk.main()
    finally:
        try:
            pipeline.set_state(Gst.State.NULL)
            pipeline.get_state(Gst.CLOCK_TIME_NONE)
        except Exception:
            pass
        try:
            SRM.cleanup()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())