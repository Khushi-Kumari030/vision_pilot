import cv2
import numpy as np

# ----------------------------
# Load image
# ----------------------------
img = cv2.imread("test_1_first_frame.jpg")

H_img, W_img = img.shape[:2]

# ----------------------------
# Camera intrinsics
# ----------------------------
fx = 360.8899
fy = 360.2564
cx = 418.3760
cy = 235.9731

# ----------------------------
# Camera setup
# ----------------------------
camera_height = 1.0          # meters
pitch_deg = 18.5696
pitch = np.radians(-pitch_deg)

# ----------------------------
# BEV parameters
# ----------------------------
bev_width = 600
bev_height = 800

# meters represented
x_min = -4.0      # left
x_max = 4.0       # right

z_min = 1.0       # near
z_max = 25.0      # far

bev = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)

# ----------------------------
# IPM
# ----------------------------
for v_bev in range(bev_height):

    z = z_max - (v_bev/(bev_height-1))*(z_max-z_min)

    for u_bev in range(bev_width):

        x = x_min + (u_bev/(bev_width-1))*(x_max-x_min)

        # Ground point in camera coordinates
        Xg = x
        Yg = 0
        Zg = z

        # Rotate by pitch
        Xc = Xg

        Yc = (
            -camera_height*np.cos(pitch)
            - Zg*np.sin(pitch)
        )

        Zc = (
            Zg*np.cos(pitch)
            - camera_height*np.sin(pitch)
        )

        if Zc <= 0:
            continue

        # Project to image
        u = fx*(Xc/Zc) + cx
        v = fy*(Yc/Zc) + cy

        u = int(round(u))
        v = int(round(v))

        if (
            0 <= u < W_img and
            0 <= v < H_img
        ):
            bev[v_bev, u_bev] = img[v, u]

# ----------------------------
# Display
# ----------------------------
cv2.imshow("Original", img)
cv2.imshow("Metric IPM", bev)

cv2.waitKey(0)
cv2.destroyAllWindows()