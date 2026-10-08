import sys
import cv2
from PIL import Image

# Add repo root
sys.path.append(r"D:\Vision_pilot\autoware_vision_pilot")

from Models.visualizations.AutoSpeed.video_visualization import AutoSpeedONNXInfer

MODEL_PATH = r"D:\Vision_pilot\autoware_vision_pilot\VisionPilot\software_defined_vehicle\OpenADKit\AutoSpeed\model-weights\autospeed.onnx"

VIDEO_PATH = r"D:\Vision_pilot\autoware_vision_pilot\VisionPilot\software_defined_vehicle\OpenADKit\Test\test_2.mp4"

model = AutoSpeedONNXInfer(MODEL_PATH)

model.train_size = (640, 640)

print("Using train size:", model.train_size)

cap = cv2.VideoCapture(VIDEO_PATH)

frame_id = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_id += 1

    # Every 30th frame
    if frame_id % 30 != 0:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    preds = model.inference(image)

    print("\n======================")
    print("FRAME:", frame_id)
    print("======================")

    print("TOTAL DETECTIONS:", len(preds))

    for det in preds:

        x1, y1, x2, y2, conf, cls = det

        print(
            f"class={int(cls)} "
            f"conf={conf:.2f} "
            f"bbox=({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
        )

cap.release()