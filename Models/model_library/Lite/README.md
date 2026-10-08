# Lite Models – Lightweight Perception Variants

All Lite models share the same architectural foundation:

> **EfficientNet-B0/B1 encoder + DeepLabV3+ decoder + lightweight heads**

The models are optimized with reduced decoder configurations and lightweight heads to significantly decrease computational cost while preserving spatial resolution and deployment flexibility (dynamic input resolution supported).

SceneSegLite and Scene3DLite use an EfficientNet-B1 encoder, while EgoLanesLite uses an EfficientNet-B0 encoder to further reduce computational cost.

---

# SceneSegLite

SceneSegLite is a lightweight semantic segmentation network performing **Cityscapes-style semantic classification (19 standard classes)**.

The 19 classes are grouped into the following categories:

### Flat Surfaces
- Road  
- Sidewalk  

### Construction
- Building  
- Wall  
- Fence  

### Objects
- Pole  
- Traffic Light  
- Traffic Sign  

### Nature
- Vegetation  
- Terrain  

### Sky
- Sky  

### Humans
- Person  
- Rider  

### Vehicles
- Car  
- Truck  
- Bus  
- Train  
- Motorcycle  
- Bicycle  


---

## Inference performance – Embedded Inference – Jetson Orin Nano (8GB)

**Forward pass includes:** Host→Device (H2D) + Inference + Device→Host (D2H)

### SceneSeg vs SceneSegLite

### SceneSeg vs SceneSegLite

| Model         | GOPs | Backend        | Forward [ms] | FPS  |
|--------------|------|---------------|--------------|------|
| SceneSeg     | 224  | FP32 ONNX     | 159.6        | 6.3  |
| SceneSeg     | 224  | FP32 TensorRT | 98.1         | 10.2 |
| SceneSegLite | 7.82 | FP32 ONNX     | 43.5         | 23.0 |
| SceneSegLite | 7.82 | FP32 TensorRT | 23.9         | 41.8 |
| SceneSegLite | 7.82 | INT8 TensorRT | 11.4         | 87.6 |


---

## SceneSegLite model weights

### [Download PyTorch Weights (*.pth)](https://drive.google.com/file/d/1nxOg-Bl5f5PVm90YebMUnu-wKWoDXG-n/view?usp=drive_link)
### [Download ONNX FP32 (*.onnx)](https://drive.google.com/file/d/1MqDNmaHz354dH9xQ2W5NT9V7P8rE9HgA/view?usp=drive_link)

---

# Scene3DLite

Scene3DLite is a lightweight monocular **relative depth estimation** model distilled from large-scale transformer-based depth models (DepthAnythingV2 as in the original Scene3D) to a compact convolutional architecture.

It preserves:

- Edge sharpness  
- Relative depth ordering  
- Scale-consistent predictions  

while reducing computational overhead.

---

## Inference performance – Embedded Inference – Jetson Orin Nano (8GB)

**Forward pass includes:** H2D + Inference + D2H

### Scene3D vs Scene3DLite

### Scene3D vs Scene3DLite

| Model         | GOPs | Backend        | Forward [ms] | FPS  |
|--------------|------|---------------|--------------|------|
| Scene3D      | 224  | FP32 ONNX     | 168.4        | 5.9  |
| Scene3D      | 224  | FP32 TensorRT | 99.7         | 10.0 |
| Scene3DLite  | 7.78 | FP32 ONNX     | 41.0         | 24.4 |
| Scene3DLite  | 7.78 | FP32 TensorRT | 23.4         | 42.7 |
| Scene3DLite  | 7.78 | INT8 TensorRT | 10.9         | 91.4 |


---

## Scene3DLite model weights

### [Download PyTorch Weights (*.pth)](https://drive.google.com/file/d/18YFLGQp0xPmhys-9SFW4BLu91PcOVXlL/view?usp=drive_link)
### [Download ONNX FP32 (*.onnx)](https://drive.google.com/file/d/1H3EUps_v_ydTnwxATsfHy_DABGt9ztiT/view?usp=drive_link)
---

# EgoLanesLite

EgoLanesLite is a lightweight **three-class lane segmentation** network.

Classes:

- `Leftmost Lane`
- `Rightmost Lane`
- `Other Lanes`

The model's output maintain the same spatial resolution as the input, instead of the original EgoLanes model which outputs at 1/4 resolution. 

---

## Validation Performance (640×320 input)
To allow for a fair comparison against EgoLanes baseline, we report the metrics of the same model trained with output stride = 1/4, like EgoLanes.

| Dataset     | Metric | EgoLanes | EgoLanesLite (OS = 1/4) | EgoLanesLite (OS = 1) |
|------------|--------|----------|--------------|--------------|
| CurveLanes | mIoU   | 46.9     | 44.6         |   24.6    |
| TuSimple   | mIoU   | 43.6     | 51.4         |   24.0    |

---

## Inference performance – Embedded Inference – Jetson Orin Nano (8GB)

**Forward pass includes:** H2D + Inference + D2H

### EgoLanes vs EgoLanesLite

| Model         | GOPs | Backend        | Forward [ms] | FPS  |
|--------------|------|---------------|--------------|------|
| EgoLanes     | 119  | FP32 ONNX     | 94.9         | 10.5 |
| EgoLanes     | 119  | FP32 TensorRT | 48.8         | 20.5 |
| EgoLanesLite (OS = 1) | 6.10 | FP32 ONNX     | 26.7         | 37.4 |
| EgoLanesLite (OS = 1) | 6.10 | FP32 TensorRT | 15.3         | 65.3 |
| EgoLanesLite (OS = 1) | 6.10 | INT8 TensorRT | 9.6         | 104.3 |


---

## EgoLanesLite model weights

### [Download PyTorch Weights (*.pth)](https://drive.google.com/file/d/1axDNIWclpTUr4U7-J656KHOCY5d2gTpV/view?usp=drive_link)
### [Download ONNX FP32 (*.onnx)](https://drive.google.com/file/d/11dXmzHryzwyFYekUTr-e_XKm-lH4Wkr3/view?usp=drive_link)