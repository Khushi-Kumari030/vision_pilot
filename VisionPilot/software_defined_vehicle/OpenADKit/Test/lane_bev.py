import cv2
import numpy as np

img = cv2.imread("frame_50.jpg")

h, w = img.shape[:2]

sx = w / 160
sy = h / 80

src = np.float32([
    [360,340],   # TL
    [560,340],   # TR
    [780,470],   # BR
    [260,470]    # BL
])

dst = np.float32([
    [150,0],
    [700,0],
    [700,480],
    [150,480]
])

debug = img.copy()

cv2.polylines(
    debug,
    [src.astype(np.int32)],
    True,
    (0,255,0),
    3
)

cv2.imwrite("roi_debug.jpg", debug)
H = cv2.getPerspectiveTransform(src, dst)
bev = cv2.warpPerspective(img, H, (848,480))

cv2.imwrite("lane_bev.jpg", bev)

cv2.imshow("Original", img)
cv2.imshow("Lane BEV", bev)

cv2.waitKey(0)
cv2.destroyAllWindows()