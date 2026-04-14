import cv2
import numpy as np
import glob

# 1. 카메라 캘리브레이션 데이터 로드 (과제 #3 결과물)
# 직접 숫자를 입력하거나 저장된 파일을 불러오세요.
# 일반적인 노트북 웹캠(720p) 표준값입니다.
mtx = np.array([[900.0, 0, 640.0], 
                [0, 900.0, 360.0], 
                [0, 0, 1.0]])
dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])              # 왜곡 계수

# 2. AR로 띄울 3D 객체 정의 (예: 3D 정육면체)
# 체스판의 한 칸 크기를 1로 잡았을 때의 좌표입니다.
# 예제와 차별화를 위해 좌표를 수정해 보세요!
def draw_cube(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # 바닥면 그리기 (녹색)
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -1)
    # 기둥 그리기
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)
    # 윗면 그리기 (빨간색)
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img

# 큐브의 8개 정점 (x, y, z)
axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3]])

# 3. 체스판 감지 및 포즈 추정
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret == True:
        # 포즈 추정 (PnP 알고리즘)
        # rvec: 회전 벡터, tvec: 이동 벡터
        ret, rvecs, tvecs = cv2.solvePnP(objp, corners, mtx, dist)

        # 3D 점을 2D 이미지 좌표로 투영
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        # 화면에 그리기
        frame = draw_cube(frame, imgpts)

    cv2.imshow('AR Pose Estimation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
