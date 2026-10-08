import sys

sys.path.append(
    r"D:\Vision_pilot\autoware_vision_pilot"
)

import cv2
from PIL import Image

# Reuse AutoSpeed ONNX class
from Models.visualizations.AutoSpeed.video_visualization import AutoSpeedONNXInfer

MODEL_PATH = r"D:\Vision_pilot\autoware_vision_pilot\VisionPilot\software_defined_vehicle\OpenADKit\AutoSpeed\model-weights\autospeed.onnx"

VIDEO_PATH = r"D:\Vision_pilot\autoware_vision_pilot\VisionPilot\software_defined_vehicle\OpenADKit\Test\test_2.mp4"

model = AutoSpeedONNXInfer(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

frame_id = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_id += 1

    # Process every 30th frame
    if frame_id % 30 != 0:
        continue

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    preds = model.inference(image_pil)

    print("\n====================")
    print("FRAME:", frame_id)
    print("====================")

    print("Detections:", len(preds))

    for p in preds:

        x1, y1, x2, y2, conf, cls = p

        print(
            f"class={int(cls)} "
            f"conf={conf:.2f} "
            f"bbox=({int(x1)},{int(y1)},{int(x2)},{int(y2)})"
        )

cap.release()