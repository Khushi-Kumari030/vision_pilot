import cv2
import numpy as np

img = cv2.imread("frame_50.jpg")

sx = 848/160
sy = 480/80

lane0 = np.array([
(73,66),(72,67),(71,68),(68,69),(66,70),
(61,71),(58,72),(56,73),(57,74),(55,75),
(53,76),(49,77),(46,78),(45,79)
])

lane1 = np.array([
(114,66),(105,67),(107,68),(122,69),(125,70),
(127,71),(129,72),(128,73),(130,74),(131,75),
(132,76),(133,77),(126,78),(125,79)
])

lane0[:,0] = lane0[:,0] * sx
lane0[:,1] = lane0[:,1] * sy

lane1[:,0] = lane1[:,0] * sx
lane1[:,1] = lane1[:,1] * sy

m0,c0 = np.polyfit(lane0[:,1], lane0[:,0], 1)
m1,c1 = np.polyfit(lane1[:,1], lane1[:,0], 1)

vp_y = (c1 - c0)/(m0 - m1)
vp_x = m0*vp_y + c0

print("Vanishing Point:")
print(int(vp_x), int(vp_y))

cv2.circle(img,(int(vp_x),int(vp_y)),10,(0,255,255),-1)

for m,c,color in [
    (m0,c0,(0,255,0)),
    (m1,c1,(0,0,255))
]:
    y1 = 480
    y2 = max(0,int(vp_y))

    x1 = int(m*y1 + c)
    x2 = int(m*y2 + c)

    cv2.line(img,(x1,y1),(x2,y2),color,3)

cv2.imshow("VP",img)
cv2.waitKey(0)
cv2.destroyAllWindows()