import sys
import cv2
from PIL import Image

# Add repository root
sys.path.append(r"D:\Vision_pilot\autoware_vision_pilot")

from Models.visualizations.AutoSpeed.video_visualization import AutoSpeedONNXInfer

# --------------------------------------------------
# Paths
# --------------------------------------------------

MODEL_PATH = r"D:\Vision_pilot\autoware_vision_pilot\VisionPilot\software_defined_vehicle\OpenADKit\AutoSpeed\model-weights\autospeed.onnx"

VIDEO_PATH = r"D:\Vision_pilot\autoware_vision_pilot\VisionPilot\software_defined_vehicle\OpenADKit\Test\test_2.mp4"

OUTPUT_IMAGE = "frame390_autospeed_debug.jpg"

# --------------------------------------------------
# Load model
# --------------------------------------------------

model = AutoSpeedONNXInfer(MODEL_PATH)

# IMPORTANT:
# Override incorrect size used by visualization script
model.train_size = (640, 640)

print("Using train size:", model.train_size)

# --------------------------------------------------
# Open video
# --------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

frame_id = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    frame_id += 1

    # Only process frame 390
    if frame_id != 390:
        continue

    print(f"\nProcessing frame {frame_id}")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    preds = model.inference(image)

    print(f"Detections: {len(preds)}")

    for idx, det in enumerate(preds):

        x1, y1, x2, y2, conf, cls = det

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        cls = int(cls)

        print(
            f"Detection {idx+1}: "
            f"class={cls} "
            f"conf={conf:.2f} "
            f"bbox=({x1},{y1},{x2},{y2})"
        )

        # --------------------------------
        # Class colors
        # --------------------------------

        if cls == 1:
            color = (0, 0, 255)      # red

        elif cls == 2:
            color = (0, 255, 255)    # yellow

        elif cls == 3:
            color = (255, 255, 0)    # cyan

        else:
            color = (255, 255, 255)

        # Draw box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # Label
        label = f"C{cls} {conf:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # Save image
    cv2.imwrite(OUTPUT_IMAGE, frame)

    print("\nSaved:")
    print(OUTPUT_IMAGE)

    break

cap.release()

print("\nDone.")