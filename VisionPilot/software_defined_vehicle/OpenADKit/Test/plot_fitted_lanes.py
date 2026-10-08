# plot_fitted_lanes.py

import cv2
import numpy as np

img = cv2.imread("test_1_first_frame.jpg")

sx = 848/160
sy = 480/80

left_fit = np.array([
0.0126787557,
-2.94695562,
208.096418
])

right_fit = np.array([
-0.00339083,
2.48173375,
-58.1098334
])

for y in range(61,80):

    xl = left_fit[0]*y*y + left_fit[1]*y + left_fit[2]
    xr = right_fit[0]*y*y + right_fit[1]*y + right_fit[2]

    cv2.circle(
        img,
        (int(xl*sx), int(y*sy)),
        4,
        (0,255,0),
        -1
    )

    cv2.circle(
        img,
        (int(xr*sx), int(y*sy)),
        4,
        (0,0,255),
        -1
    )

cv2.imshow("Lane Fit",img)
cv2.waitKey(0)
cv2.destroyAllWindows()