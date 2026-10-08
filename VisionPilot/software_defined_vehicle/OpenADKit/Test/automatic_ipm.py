import cv2
import numpy as np

img = cv2.imread("test_1_first_frame.jpg")

vp_x = 411
vp_y = 357

src = np.float32([
    [vp_x-25, vp_y],
    [vp_x+25, vp_y],
    [760,479],
    [120,479]
])

dst = np.float32([
    [250,0],
    [600,0],
    [600,480],
    [250,480]
])

H = cv2.getPerspectiveTransform(src,dst)

bev = cv2.warpPerspective(
    img,
    H,
    (850,480)
)

cv2.imshow("Original",img)
cv2.imshow("BEV",bev)

cv2.waitKey(0)
cv2.destroyAllWindows()