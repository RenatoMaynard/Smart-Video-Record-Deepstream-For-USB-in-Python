# Smart-Video-Record-Deepstream-For-USB-in-Python
Guide to set up DeepStream pyds and run an USB pipeline in Python. 

# Single USB Camera DeepStream Demo (Person-Only + Smart Recording)

This repository contains a **minimal, single-camera** DeepStream + GStreamer example designed for **NVIDIA Jetson**.
It is intentionally “clean-room simple” so you can open it in **Visual Studio Code**, press **Run**, and immediately see
how a DeepStream pipeline is built and how **Smart Recording** is triggered.

What it does:
- Uses **one USB camera** (`/dev/videoX`) and displays it **full screen** .
- Runs inference with **`nvinfer`**.
- Draws **ONLY** `person` detections and their **confidence**.
- Supports **Smart Recording** (USB mode via **NvDsSR** / DeepStream smartrecord library).
- Includes a manual hotkey (R) to record a clip.

---

## Files you should upload

- `usb_cam.py`  
  Main application: camera capture → inference → OSD → fullscreen display → Smart Record triggers.

- `usb_smartrec.py`  
  Smart Recording helper module (manual NvDsSR for USB).

---

## Background: how this demo is built

DeepStream applications are typically GStreamer pipelines with NVIDIA-accelerated elements.
This demo is intentionally built “the long way” (explicitly creating and linking elements) so people can learn from it.

### 1) Video capture and decode (USB)

USB cameras commonly output MJPEG. The demo requests MJPEG from the camera and decodes it with NVIDIA hardware:

**Pipeline concept (simplified):**
- `v4l2src`  
  Reads frames from `/dev/videoX`.
- `capsfilter` (MJPEG caps)  
  Asks the camera for `image/jpeg` at a chosen resolution / FPS.
- `nvv4l2decoder`  
  Hardware decoder (MJPEG → NV12 in GPU memory).
- `nvvidconv` / `nvvideoconvert`  
  Converts to the format DeepStream components expect.

### 2) Inference (nvinfer)

DeepStream inference is done by `nvinfer`, configured by a `.txt` file that points to:
- the model/engine
- label file (optional)
- preprocessing settings
- class settings


### 3) On-screen display (nvdsosd)

`nvdsosd` draws boxes and text.  
This demo modifies metadata so it only draws:
- objects that are “person” (class-id match)
- text label showing confidence

Everything else is hidden.

### 4) Smart Recording (USB)

DeepStream has two Smart Record approaches:

1. **Built-in Smart Record inside `nvurisrcbin`**  
   Great for RTSP sources. (Not used here.)

2. **Manual Smart Record (NvDsSR)**  
   This is needed for a USB pipeline. It requires DeepStream’s smartrecord library:
- `libnvdsgst_smartrecord.so`

In this demo:
- We split the stream with a `tee`.
- One branch goes to display/inference.
- Another branch feeds an **encoder + parser** into the **NvDsSR recordbin**.

**Recording concept:**
- Smart Record keeps a small circular buffer (“cache”) so you can save **a few seconds before** the trigger.
- When triggered, we record:
  - `SR_BACK_SEC` seconds in the past
  - `SR_FRONT_SEC` seconds into the future

This is exactly how “event recording” works in many camera systems.

### 5) Triggers

The demo supports two triggers:
- **Automatic:** when a person is detected above `MIN_PERSON_CONF`
- **Manual:** press **R** to record a clip

---

## Requirements

This demo is meant for **Jetson + DeepStream**.

You should already have:
- NVIDIA DeepStream installed
- DeepStream Python bindings (`pyds`) working
- PyGObject (`gi`) working
- A USB camera available as `/dev/videoX`

### Smart Recording requirement (USB)
For USB Smart Recording, DeepStream’s smartrecord library must be accessible.
Most commonly it exists at:

- `/opt/nvidia/deepstream/deepstream/lib/libnvdsgst_smartrecord.so`

If the library is missing, the app can still run, but **Smart Recording will be disabled**.

---

## How to run (Visual Studio Code)

1. Open the folder in **VS Code** on the Jetson.
2. Open `usb_cam.py`.
3. Edit the configuration block at the top (next section).
4. Run.

No command line arguments are required.

---

## Configuration (edit at top of `usb_cam_demo_single_persononly.py`)

- **USB device**
```python
USB_DEVICE = "/dev/video0"
```

- **PGIE config**
```python
PGIE_CONFIG = "/home/nvidia/Desktop/new/dstest1_pgie_config.txt"
```

- **Person class IDs**
```python
PERSON_CLASS_IDS = {0}
```
If your model uses a different class id for person, change it here.

- **Minimum confidence**
```python
MIN_PERSON_CONF = 0.35
```

- **Smart Record clip duration**
```python
SR_BACK_SEC = 10
SR_FRONT_SEC = 10
```

- **Output folder**
```python
SMARTREC_DIR = "SmartRecDir"
```

---

## Controls

- **R** → Start a Smart Record clip immediately
- **ESC** → Exit

---

## Output clips

Recordings are saved under:
- `SmartRecDir/`

The filenames include a prefix and timestamp (depending on the DeepStream SR backend).

---

## Troubleshooting

### `NVDSINFER_CONFIG_FAILED`
Almost always means something inside the PGIE config is wrong:
- bad path to the config file itself
- model/engine file not found
- label file not found
- invalid config option for your DeepStream version

Fix: open the PGIE config and verify every referenced path exists.

### Smart Recording doesn’t work (USB)
Smart Recording for USB requires DeepStream’s smartrecord library to be present.
If it’s missing, recording will be disabled.

### No video frames
Double-check:
- the correct `/dev/videoX` device
- camera permissions
- camera supports the requested caps (resolution/FPS)

---

## Notes for people forking this repo

- This is a learning-oriented example, not a full product.
- It’s intentionally small and readable.

---

## License

MIT
