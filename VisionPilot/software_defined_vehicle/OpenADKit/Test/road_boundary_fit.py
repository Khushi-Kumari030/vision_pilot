import cv2
import numpy as np

img = cv2.imread("frame_50.jpg")

sx = 848 / 160
sy = 480 / 80

# LEFT ROAD BOUNDARY (Lane 0)
lane0 = np.array([
    (73,66),(72,67),(71,68),(68,69),(66,70),
    (61,71),(58,72),(56,73),(57,74),(55,75),
    (53,76),(49,77),(46,78),(45,79)
], dtype=np.float32)

lane0[:,0] *= sx
lane0[:,1] *= sy

# MANUAL RIGHT ROAD BOUNDARY
right_boundary = np.array([
    (840,387),
    (818,385),
    (794,384),
    (769,383),
    (750,383),
    (728,380),
    (699,380),
    (667,379),
    (637,376),
    (612,377),
    (592,372),
    (571,374)
], dtype=np.float32)

# Draw raw points
for x,y in lane0:
    cv2.circle(img,(int(x),int(y)),5,(0,255,0),-1)

for x,y in right_boundary:
    cv2.circle(img,(int(x),int(y)),5,(0,0,255),-1)

# Fit left boundary
m0, c0 = np.polyfit(
    lane0[:,1],
    lane0[:,0],
    1
)

# Fit right boundary
m1, c1 = np.polyfit(
    right_boundary[:,1],
    right_boundary[:,0],
    1
)

# Vanishing Point
vp_y = (c1 - c0) / (m0 - m1)
vp_x = m0 * vp_y + c0

print("\nVanishing Point:")
print(int(vp_x), int(vp_y))

# Draw fitted lines
y_bottom = 479
y_top = max(0, int(vp_y))

# Left line
x0_bottom = int(m0 * y_bottom + c0)
x0_top = int(m0 * y_top + c0)

# Right line
x1_bottom = int(m1 * y_bottom + c1)
x1_top = int(m1 * y_top + c1)

cv2.line(
    img,
    (x0_bottom, y_bottom),
    (x0_top, y_top),
    (0,255,0),
    4
)

cv2.line(
    img,
    (x1_bottom, y_bottom),
    (x1_top, y_top),
    (0,0,255),
    4
)

# Draw VP
cv2.circle(
    img,
    (int(vp_x), int(vp_y)),
    10,
    (0,255,255),
    -1
)

# Generate automatic road ROI
left_bottom  = (x0_bottom, y_bottom)
right_bottom = (x1_bottom, y_bottom)

left_top  = (x0_top, y_top)
right_top = (x1_top, y_top)

pts = np.array([
    left_bottom,
    right_bottom,
    right_top,
    left_top
], np.int32)

cv2.polylines(
    img,
    [pts],
    True,
    (255,255,0),
    3
)

print("\nROI Points:")
print("Left Bottom :", left_bottom)
print("Right Bottom:", right_bottom)
print("Right Top   :", right_top)
print("Left Top    :", left_top)

cv2.imshow("Road ROI", img)
cv2.waitKey(0)
cv2.destroyAllWindows()