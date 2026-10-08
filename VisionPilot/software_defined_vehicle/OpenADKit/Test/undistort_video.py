import cv2
import numpy as np

K = np.array([
    [360.88993116, 0.0, 418.37599341],
    [0.0, 360.25640768, 235.97305177],
    [0.0, 0.0, 1.0]
])

dist = np.array([
    -0.16611822,
     0.05135685,
     0.00073602,
    -0.00209655,
    -0.01043283
])

input_video = "test_1.mp4"
output_video = "test_1_undistorted.avi"

cap = cv2.VideoCapture(input_video)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    undistorted = cv2.undistort(frame, K, dist)

    out.write(undistorted)

cap.release()
out.release()

print("Saved:", output_video)