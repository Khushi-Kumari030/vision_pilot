import cv2
import numpy as np

img = cv2.imread("frame_50.jpg")

pts = np.array([
    [260, 470],   # bottom left
    [780, 470],   # bottom right
    [560, 340],   # top right
    [360, 340]    # top left
], np.int32)

cv2.polylines(img, [pts], True, (0,255,0), 3)

for i, (x,y) in enumerate(pts):
    cv2.circle(img, (x,y), 6, (0,0,255), -1)
    cv2.putText(img, str(i+1), (x+10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

cv2.imshow("Trapezoid", img)
cv2.waitKey(0)
cv2.destroyAllWindows()