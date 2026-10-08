import cv2
import numpy as np

img = cv2.imread("frame_50.jpg")

sx = 848/160
sy = 480/80

lane0 = np.array([
(73,66),(72,67),(71,68),(68,69),(66,70),
(61,71),(58,72),(56,73),(57,74),(55,75),
(53,76),(49,77),(46,78),(45,79)
], dtype=np.float32)

lane1 = np.array([
(114,66),(105,67),(107,68),(122,69),(125,70),
(127,71),(129,72),(128,73),(130,74),(131,75),
(132,76),(133,77),(126,78),(125,79)
], dtype=np.float32)

lane0[:,0] *= sx
lane0[:,1] *= sy

lane1[:,0] *= sx
lane1[:,1] *= sy

# draw original lanes
for x,y in lane0:
    cv2.circle(img,(int(x),int(y)),5,(0,255,0),-1)

for x,y in lane1:
    cv2.circle(img,(int(x),int(y)),5,(0,0,255),-1)

# synthetic right boundary
lane2 = []

for p0,p1 in zip(lane0,lane1):

    width = p1[0] - p0[0]

    x_new = p1[0] + width
    y_new = p1[1]

    lane2.append([x_new,y_new])

lane2 = np.array(lane2)

# draw synthetic boundary
for x,y in lane2:
    cv2.circle(img,(int(x),int(y)),5,(255,0,0),-1)

cv2.imshow("Synthetic Boundary",img)
cv2.waitKey(0)
cv2.destroyAllWindows()