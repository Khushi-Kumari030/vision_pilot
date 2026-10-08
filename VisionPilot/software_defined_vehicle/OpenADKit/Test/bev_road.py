import cv2
import numpy as np

img = cv2.imread("test_1_first_frame.jpg")

src = np.float32([
    [269,455],
    [838,446],
    [382,388],
    [568,385]
])

dst = np.float32([
    [250,480],
    [600,480],
    [250,0],
    [600,0]
])

H = cv2.getPerspectiveTransform(src, dst)

bev = cv2.warpPerspective(img, H, (848,480))

cv2.imwrite("bev_output.jpg", bev)

cv2.imshow("Original", img)
cv2.imshow("BEV", bev)

cv2.waitKey(0)
cv2.destroyAllWindows()