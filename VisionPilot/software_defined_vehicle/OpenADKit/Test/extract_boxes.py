import cv2
import numpy as np

VIDEO = "test_2_autospeed.avi"
OUTPUT_VIDEO = "filtered_detections.avi"

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    print("Could not open video!")
    exit()

# ===============================
# Video Writer
# ===============================
fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

frame_number = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    display = frame.copy()

    # ===============================
    # Convert to HSV
    # ===============================
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Cyan color range (AutoSpeed boxes)
    lower = np.array([75, 50, 50])
    upper = np.array([105, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # ===============================
    # Morphological cleaning
    # ===============================
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # ===============================
    # Find contours
    # ===============================
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    vehicle_count = 0

    # ===============================
    # Process detections
    # ===============================
    for cnt in contours:

        x, y, w, h = cv2.boundingRect(cnt)

        # Ignore detections above ROI
        if y < 250:
            continue

        # Size filtering
        if w < 30 or h < 25:
            continue

        if w > 250 or h > 250:
            continue

        # Aspect ratio filtering
        aspect = w / h

        if aspect < 0.5 or aspect > 3.0:
            continue

        vehicle_count += 1

        # Draw green box
        cv2.rectangle(
            display,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Bottom-center point
        u = x + w // 2
        v = y + h

        cv2.circle(
            display,
            (u, v),
            5,
            (0, 0, 255),
            -1
        )

        # Vehicle label
        cv2.putText(
            display,
            f"V{vehicle_count}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # Print details
        print(
            f"Frame {frame_number} | "
            f"Vehicle {vehicle_count}: "
            f"bbox=({x},{y},{x+w},{y+h}), "
            f"bottom-center=({u},{v})"
        )

    # ===============================
    # Vehicle Count
    # ===============================
    cv2.putText(
        display,
        f"Count: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # ===============================
    # ROI Line
    # ===============================
    cv2.line(
        display,
        (0, 250),
        (width, 250),
        (255, 0, 255),
        2
    )

    cv2.putText(
        display,
        "ROI",
        (10, 245),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2
    )

    # ===============================
    # Save Output Video
    # ===============================
    out.write(display)

    # ===============================
    # Display
    # ===============================
    cv2.imshow("Extracted Vehicles", display)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)

    # ESC to quit
    if key == 27:
        break

# ===============================
# Cleanup
# ===============================
cap.release()
out.release()

cv2.destroyAllWindows()

print("\nSaved output video as:")
print(OUTPUT_VIDEO)