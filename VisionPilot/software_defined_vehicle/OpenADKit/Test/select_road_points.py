import cv2

img = cv2.imread("test_1_first_frame.jpg")

pts = []

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))
        print(len(pts), (x, y))

while True:
    cv2.imshow("Road", img)
    cv2.setMouseCallback("Road", click)

    key = cv2.waitKey(1)

    if key == 27:
        break

cv2.destroyAllWindows()

print("\nPoints:")
print(pts)