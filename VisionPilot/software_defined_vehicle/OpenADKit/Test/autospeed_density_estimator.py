import sys
import cv2
from PIL import Image

# --------------------------------------------------
# Add repo root
# --------------------------------------------------

sys.path.append(r"D:\Vision_pilot\autoware_vision_pilot")

from Models.visualizations.AutoSpeed.video_visualization import AutoSpeedONNXInfer

# --------------------------------------------------
# Paths
# --------------------------------------------------

MODEL_PATH = r"D:\Vision_pilot\autoware_vision_pilot\VisionPilot\software_defined_vehicle\OpenADKit\AutoSpeed\model-weights\autospeed.onnx"

VIDEO_PATH = r"D:\Vision_pilot\autoware_vision_pilot\VisionPilot\software_defined_vehicle\OpenADKit\Test\test_2.mp4"

OUTPUT_VIDEO = r"D:\Vision_pilot\autoware_vision_pilot\VisionPilot\software_defined_vehicle\OpenADKit\Test\test_2_density.avi"

# --------------------------------------------------
# Density settings
# --------------------------------------------------

VISIBLE_ROAD_LENGTH_METERS = 20.0
MAX_DENSITY = 150.0

MIN_BOX_WIDTH = 25
MIN_BOX_HEIGHT = 25

# --------------------------------------------------
# Load model
# --------------------------------------------------

model = AutoSpeedONNXInfer(MODEL_PATH)

# ONNX expects 640x640
model.train_size = (640, 640)

# --------------------------------------------------
# Open video
# --------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*'XVID'),
    fps,
    (width, height)
)

frame_id = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_id += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    preds = model.inference(image)

    vehicle_count = 0

    for det in preds:

        x1, y1, x2, y2, conf, cls = det

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        w = x2 - x1
        h = y2 - y1

        # Filter tiny detections
        if w < MIN_BOX_WIDTH:
            continue

        if h < MIN_BOX_HEIGHT:
            continue

        vehicle_count += 1

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"V{vehicle_count}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # ----------------------------------
    # Density Calculation
    # ----------------------------------

    density = vehicle_count / (VISIBLE_ROAD_LENGTH_METERS / 1000.0)

    capacity_pct = min(
        density / MAX_DENSITY * 100.0,
        100.0
    )

    if capacity_pct < 35:
        status = "LIGHT"

    elif capacity_pct < 70:
        status = "MODERATE"

    else:
        status = "HEAVY"

    # ----------------------------------
    # Overlay
    # ----------------------------------

    cv2.rectangle(
        frame,
        (10, 10),
        (420, 140),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Density: {density:.1f} veh/km",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Capacity: {capacity_pct:.1f}%",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Status: {status}",
        (220, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if status == "LIGHT"
        else (0, 255, 255) if status == "MODERATE"
        else (0, 0, 255),
        2
    )

    writer.write(frame)

    cv2.imshow("Density Estimator", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print()
print("Saved:")
print(OUTPUT_VIDEO)