import cv2
import os

out_dir = "camera_check"
os.makedirs(out_dir, exist_ok=True)

for i in range(2):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"Camera index {i}: not opened")
        continue

    # 少し設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 最初の数フレームは捨てる
    frame = None
    for _ in range(10):
        ret, frame = cap.read()

    if ret and frame is not None:
        path = os.path.join(out_dir, f"camera_{i}.jpg")
        cv2.imwrite(path, frame)
        print(f"Camera index {i}: saved {path}, shape={frame.shape}")
    else:
        print(f"Camera index {i}: opened but no frame")

    cap.release()