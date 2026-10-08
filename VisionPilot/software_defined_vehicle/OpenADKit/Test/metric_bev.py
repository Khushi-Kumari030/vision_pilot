import cv2
import numpy as np

# =========================
# Load undistorted image
# =========================
img = cv2.imread("test_1_first_frame.jpg")

h_img, w_img = img.shape[:2]

# =========================
# Camera Intrinsics
# =========================
K = np.array([
    [360.8899, 0, 418.3760],
    [0, 360.2564, 235.9731],
    [0, 0, 1]
], dtype=np.float32)

D = np.array([
    -0.16611822,
     0.05135685,
     0.00073602,
    -0.00209655,
    -0.01043283
], dtype=np.float32)

# =========================
# Camera Pose
# =========================
height = 1.0

pitch_deg = 18.569597431099865
yaw_deg = -1.17

pitch = np.radians(pitch_deg)
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

tvec = np.array([
    0,
    height,
    0
], dtype=np.float32)

# =========================
# Metric BEV Parameters
# =========================
x_min = -5.0      # metres
x_max = 5.0

z_min = 0.0
z_max = 30.0

resolution = 0.05   # metres/pixel

bev_width = int((x_max - x_min) / resolution)
bev_height = int((z_max - z_min) / resolution)

bev = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)

# =========================
# Generate BEV
# =========================
for row in range(bev_height):

    # Forward distance
    z = z_max - row * resolution

    for col in range(bev_width):

        # Lateral distance
        x = x_min + col * resolution

        point = np.array(
            [[x, 0, z]],
            dtype=np.float32
        )

        imgpts, _ = cv2.projectPoints(
            point,
            rvec,
            tvec,
            K,
            None      # image already undistorted
        )

        u, v = imgpts[0][0]

        u = int(round(u))
        v = int(round(v))

        if (0 <= u < w_img) and (0 <= v < h_img):

            bev[row, col] = img[v, u]

# =========================
# Draw Metric Grid
# =========================
for z in range(0, 31, 5):

    r = int((z_max - z) / resolution)

    cv2.line(
        bev,
        (0, r),
        (bev_width, r),
        (255,255,255),
        1
    )

    cv2.putText(
        bev,
        f"{z}m",
        (5, max(r-5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255,255,255),
        1
    )

# Center line
center_col = bev_width // 2

cv2.line(
    bev,
    (center_col, 0),
    (center_col, bev_height),
    (0,255,255),
    1
)

# Ego vehicle marker
cv2.circle(
    bev,
    (center_col, bev_height-1),
    5,
    (0,0,255),
    -1
)

# =========================
# Display
# =========================
cv2.imshow("Original", img)
cv2.imshow("Metric BEV", bev)

cv2.waitKey(0)
cv2.destroyAllWindows()