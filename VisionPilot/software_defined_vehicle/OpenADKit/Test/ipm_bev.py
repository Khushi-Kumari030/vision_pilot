import cv2
import numpy as np

img = cv2.imread("frame_50.jpg")

h, w = img.shape[:2]

# Assumed camera parameters
camera_height = 1.0      # meters
pitch_deg = 8.0          # assumed downward pitch

# Approximate focal lengths from FOV
fov_x = 120.0

fx = w / (2 * np.tan(np.radians(fov_x / 2)))
fy = fx

cx = w / 2
cy = h / 2

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
])

pitch = np.radians(pitch_deg)

R = np.array([
    [1, 0, 0],
    [0, np.cos(pitch), -np.sin(pitch)],
    [0, np.sin(pitch),  np.cos(pitch)]
])

T = np.array([[0], [camera_height], [0]])

# Simple road trapezoid
src = np.float32([
    [200, h-1],
    [w-200, h-1],
    [380, 250],
    [470, 250]
])

dst = np.float32([
    [250, h],
    [600, h],
    [250, 0],
    [600, 0]
])

H = cv2.getPerspectiveTransform(src, dst)

bev = cv2.warpPerspective(img, H, (w, h))

cv2.imwrite("frame50_bev.jpg", bev)

cv2.imshow("Original", img)
cv2.imshow("BEV", bev)

cv2.waitKey(0)
cv2.destroyAllWindows()