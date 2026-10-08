# roi_from_vp.py

import cv2
import numpy as np

img = cv2.imread("test_1_first_frame.jpg")

vp_x = 411
vp_y = 357

bottom_y = 479

src = np.float32([
    [vp_x - 60, vp_y],     # TL
    [vp_x + 60, vp_y],     # TR
    [760, bottom_y],       # BR
    [120, bottom_y]        # BL
])

pts = src.astype(np.int32)

cv2.polylines(
    img,
    [pts],
    True,
    (0,255,0),
    3
)

cv2.circle(
    img,
    (vp_x,vp_y),
    8,
    (0,255,255),
    -1
)

cv2.imshow("ROI",img)
cv2.waitKey(0)
cv2.destroyAllWindows()