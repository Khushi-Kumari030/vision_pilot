import cv2

img = cv2.imread("frame_50.jpg")

points = []

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x,y))
        print((x,y))

        cv2.circle(img, (x,y), 5, (0,0,255), -1)
        cv2.imshow("Image", img)

cv2.imshow("Image", img)
cv2.setMouseCallback("Image", click)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nAll Points:")
print(points)