import cv2
import numpy as np

img = cv2.imread("test_1_first_frame.jpg")

# Camera Matrix
K = np.array([
    [360.8899, 0, 418.3760],
    [0, 360.2564, 235.9731],
    [0, 0, 1]
], dtype=np.float32)

# Distortion coefficients
D = np.array([
    -0.16611822,
     0.05135685,
     0.00073602,
    -0.00209655,
    -0.01043283
], dtype=np.float32)

# Pitch (downward)
pitch_deg = 18.569597431099865
pitch = np.radians(pitch_deg)

# Rotation vector (rotate about X-axis)
rvec = np.array([
    -pitch,
    0,
    0
], dtype=np.float32)

# Camera height = 1 m
tvec = np.array([
    0,
    1.0,
    0
], dtype=np.float32)

# Ground points: X=right, Y=0, Z=forward
points = np.array([
    [0, 0, 5],
    [0, 0, 10],
    [0, 0, 15],
    [0, 0, 20]
], dtype=np.float32)

imgpts, _ = cv2.projectPoints(
    points,
    rvec,
    tvec,
    K,
    D
)

print("Projected Points:")

for i, p in enumerate(imgpts):
    u, v = p.ravel()

    print(f"{points[i]} --> ({u:.2f}, {v:.2f})")

    cv2.circle(
        img,
        (int(u), int(v)),
        8,
        (0, 255, 255),
        -1
    )

    cv2.putText(
        img,
        f"{points[i][2]}m",
        (int(u)+10, int(v)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0,255,255),
        2
    )

cv2.imshow("Ground Projection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()