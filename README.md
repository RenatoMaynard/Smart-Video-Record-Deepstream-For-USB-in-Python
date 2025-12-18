# Smart-Video-Record-DeepStream-For-USB-in-Python

A practical guide + minimal code to:
- set up **DeepStream Python (`pyds`)**
- run a **single USB camera** pipeline in Python
- do **YOLO inference with `nvinfer`**
- **record smart clips** (pre/post event) from a USB source using **NvDsSR**

---

## What this demo does

- Uses **one USB camera** (`/dev/videoX`) and displays it **full screen** (no layout switching, no logo).
- Runs inference with **`nvinfer`** using a config file (example included).
- Draws **ONLY** `person` detections and their **confidence**.
- Supports **Smart Recording** for USB (manual **NvDsSR** via DeepStream smartrecord library).
- Hotkeys:
  - **R** → record a clip
  - **ESC** → exit

---

## Repository files

- `usb_cam.py`  
  Main app: USB camera → decode → streammux → nvinfer → OSD → fullscreen sink → Smart Record triggers.

- `usb_smartrec.py`  
  Smart Recording helper (manual NvDsSR wiring for USB).

- `pt_to_onnx.py`  
  Transform .pt to .onnx.
  
- `dstest1_pgie_config.txt`  
  Example `nvinfer` configuration for YOLO.  
  Key fields include `onnx-file`, `model-engine-file`, `parse-bbox-func-name`, `custom-lib-path`, and `engine-create-func-name`. 

- `labels.txt`  
  Example label file (COCO-style list). 

---

## Background: how it works (high level)

DeepStream apps are usually **GStreamer pipelines** using NVIDIA-accelerated elements.

### 1) USB capture + decode
Typical USB webcams output MJPEG. The pipeline requests MJPEG from `/dev/videoX` and decodes it using NVIDIA decode:
- `v4l2src` → reads `/dev/videoX`
- `capsfilter` → requests `image/jpeg` with a chosen resolution/FPS
- `nvv4l2decoder` → hardware MJPEG decode into NVMM memory
- `nvvidconv` / `nvvideoconvert` → color/format conversion for DeepStream elements

### 2) Inference with `nvinfer`
Inference is handled by `nvinfer`, which reads a config file.
Your sample config uses:
- `onnx-file=yolov11.pt.onnx`
- `model-engine-file=/home/nvidia/Desktop/Forklift/Engine/b1_gpu0_fp16.engine`
- `parse-bbox-func-name=NvDsInferParseYolo`
- `custom-lib-path=.../libnvdsinfer_custom_impl_Yolo.so`
- `engine-create-func-name=NvDsInferYoloCudaEngineGet` 

### 3) OSD (drawing)
`nvdsosd` draws boxes/text.
This demo **filters metadata** so it only draws:
- `person`
- confidence text like: `person 87.3%`

### 4) Smart Recording for USB (NvDsSR)
DeepStream has two Smart Record modes:

1) `nvurisrcbin` built-in Smart Record (great for RTSP)  
2) **Manual NvDsSR** (needed for USB / v4l2)

This repo uses **manual NvDsSR** so USB works:
- A `tee` splits the stream
- One branch goes to inference + display
- The other branch goes through encoder + parser into the NvDsSR recordbin (so the recorder gets **encoded** H264/H265)

Smart Record keeps a small “cache” (ring buffer), so each clip can include:
- **`SR_BACK_SEC`** seconds *before* the trigger
- **`SR_FRONT_SEC`** seconds *after* the trigger

---

# Setup / Configuration Guide (Important)

## A) Install DeepStream
Install NVIDIA DeepStream on your Jetson (via JetPack + DeepStream SDK installer, or NVIDIA packages).
After install, you should have:
- `/opt/nvidia/deepstream/deepstream/`

## B) Install DeepStream Python (`pyds`)
You must be able to run:
```python
import pyds
```
If `import pyds` fails, fix DeepStream Python first (this repo assumes it already works).

## C) Enable YOLO in DeepStream (the critical part)

DeepStream **does not natively know how to post-process YOLO outputs**.
You need a YOLO custom parser and (often) an engine creation helper. The most common solution is:

- **marcoslucianops / DeepStream-Yolo**

### 1) Build the YOLO custom parser `.so`
In the DeepStream-Yolo repo, compile the library:
```bash
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
```
That produces `libnvdsinfer_custom_impl_Yolo.so`. 

### 2) Point your `nvinfer` config to the `.so`
Your `dstest1_pgie_config.txt` already does this via:
- `custom-lib-path=/home/nvidia/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so` 

### 3) Use the YOLO parser function
Your config sets:
- `parse-bbox-func-name=NvDsInferParseYolo` 

### 4) (Optional) Engine creation helper
Your config sets:
- `engine-create-func-name=NvDsInferYoloCudaEngineGet`

This allows DeepStream-Yolo to help generate the TensorRT engine in some workflows.

---

## D) Model files (ONNX + TensorRT engine)

Your config expects:
- `onnx-file=yolov11.pt.onnx`
- `model-engine-file=/home/nvidia/Desktop/Engine/b1_gpu0_fp16.engine` 

You must ensure:
- the ONNX file exists at the path given (You can use the provided pt_to_onnx.py file)
- the engine file exists at the path given  
  (or DeepStream is allowed to generate it depending on your setup)

**Tip:** If you move files, update the config paths.

---

## E) Labels + class count (very important)

Your config currently has:
- `num-detected-classes=1` 

But your `labels.txt` contains many labels (COCO list). 

You should make these consistent:
- If your model is **person-only**, keep `num-detected-classes=1` and use a **single-line** labels file:
  - `person`
- If your model is COCO (80 classes), set:
  - `num-detected-classes=80`
  - and keep the COCO labels file

---

# Running the demo

1. Open this repo folder in **Visual Studio Code** on the Jetson.
2. Open `usb_cam.py`.
3. Edit the config block at the top:
   - `USB_DEVICE = "/dev/video0"`
   - `PGIE_CONFIG = "<absolute path>/dstest1_pgie_config.txt"`
4. Click **Run**.

No command-line args required.

---

## Controls
- **R** → record a Smart Record clip
- **ESC** → exit

---

## Output clips
Clips are written to:
- `SmartRecDir/` (default)

---

## Troubleshooting

### `NVDSINFER_CONFIG_FAILED`
Usually one of:
- bad config file path
- missing ONNX/engine file
- wrong `custom-lib-path` (YOLO parser `.so` not found)
- mismatch between model output and parser settings

### YOLO runs but no detections
Common causes:
- wrong `num-detected-classes`
- wrong thresholds (`pre-cluster-threshold`, NMS settings)
- wrong input dims / preprocessing settings for your exported ONNX

---

## License
MIT
