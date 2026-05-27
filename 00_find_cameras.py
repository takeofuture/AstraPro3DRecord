import cv2

for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Camera index {i}: OK, shape={frame.shape}")
        else:
            print(f"Camera index {i}: opened but no frame")
    cap.release()