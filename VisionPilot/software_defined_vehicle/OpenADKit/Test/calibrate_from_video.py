import cv2
import numpy as np

video_path = "checkerboard.mp4"

pattern_size = (8, 5)

objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

cap = cv2.VideoCapture(video_path)

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    if frame_id % 10 != 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCornersSB(gray, pattern_size)

    if found:
        objpoints.append(objp)
        imgpoints.append(corners)

        print(f"Frame {frame_id}: detected")

cap.release()

print("Total valid frames:", len(objpoints))

if len(objpoints) < 10:
    print("Not enough calibration frames")
    exit()

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

print("\n=== CAMERA MATRIX ===")
print(camera_matrix)

print("\n=== DISTORTION COEFFICIENTS ===")
print(dist_coeffs)

print("\n=== RMS ERROR ===")
print(ret)