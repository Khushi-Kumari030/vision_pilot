import cv2
import numpy as np

# ==========================
# Load image
# ==========================
img = cv2.imread("test_1_first_frame.jpg")

# ==========================
# Camera Matrix
# ==========================
K = np.array([
    [360.8899, 0, 418.3760],
    [0, 360.2564, 235.9731],
    [0, 0, 1]
], dtype=np.float32)

# ==========================
# Distortion Coefficients
# ==========================
D = np.array([
    -0.16611822,
     0.05135685,
     0.00073602,
    -0.00209655,
    -0.01043283
], dtype=np.float32)

# ==========================
# Camera Parameters
# ==========================
pitch_deg = 18.569597431099865
pitch = np.radians(pitch_deg)

camera_height = 1.0  # meters

# Translation vector
tvec = np.array([
    0,
    camera_height,
    0
], dtype=np.float32)

# ==========================
# Ground Points
# ==========================
# X = lateral (m)
# Y = ground plane
# Z = forward (m)

points = np.array([
    [-1.75, 0, 10],
    [ 0.00, 0, 10],
    [ 1.75, 0, 10],

    [-1.75, 0, 20],
    [ 0.00, 0, 20],
    [ 1.75, 0, 20],
], dtype=np.float32)

labels = [
    "L10", "C10", "R10",
    "L20", "C20", "R20"
]

colors = [
    (0,255,0),    # Left = Green
    (0,255,255),  # Center = Yellow
    (0,0,255),    # Right = Red

    (0,255,0),
    (0,255,255),
    (0,0,255)
]

# ==========================
# Test Multiple Yaw Values
# ==========================
yaw_values = [-1, -2, -3, -4]

for yaw_deg in yaw_values:

    test_img = img.copy()

    yaw = np.radians(yaw_deg)

    # Rotation matrices
    R_pitch = cv2.Rodrigues(
        np.array([-pitch, 0, 0], dtype=np.float32)
    )[0]

    R_yaw = cv2.Rodrigues(
        np.array([0, yaw, 0], dtype=np.float32)
    )[0]

    R = R_yaw @ R_pitch

    rvec, _ = cv2.Rodrigues(R)

    # Projection
    imgpts, _ = cv2.projectPoints(
        points,
        rvec,
        tvec,
        K,
        D
    )

    print("\n======================")
    print(f"Yaw = {yaw_deg}°")
    print("======================")

    for i, p in enumerate(imgpts):

        u, v = p.ravel()

        print(
            f"{labels[i]} : "
            f"({points[i][0]}m, {points[i][2]}m)"
            f" --> ({u:.2f}, {v:.2f})"
        )

        # Draw projected points
        cv2.circle(
            test_img,
            (int(u), int(v)),
            8,
            colors[i],
            -1
        )

        cv2.putText(
            test_img,
            labels[i],
            (int(u)+10, int(v)-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colors[i],
            2
        )

    # Display yaw value on image
    cv2.putText(
        test_img,
        f"Yaw = {yaw_deg} deg",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255,255,255),
        2
    )

    cv2.imshow(f"Yaw {yaw_deg}", test_img)

cv2.waitKey(0)
cv2.destroyAllWindows()