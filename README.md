# Camera Pose Estimation and AR

## 1. Project Overview
This project estimates the camera pose using a 9x6 chessboard and visualizes a 3D AR object on top of it.

## 2. Methodology
- **Calibration**: Used `cv2.calibrateCamera` to obtain the intrinsic matrix and distortion coefficients.
- **Pose Estimation**: Used `cv2.solvePnP` to find the rotation and translation vectors.
- **Visualization**: Projected 3D points to the 2D image plane using `cv2.projectPoints` and drew a 3D cube.

## 3. Requirements
- Python 3.x
- OpenCV (`opencv-python`)
- NumPy

## 4. How to Run
```bash
python camera_pose_ar.py

## 5. Result
![AR Demo](./result.png)
