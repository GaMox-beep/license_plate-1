import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
from ultralytics import YOLO
import easyocr
from src.utils import read_license_plate

# Khởi tạo model phát hiện biển số (YOLO)
license_plate_detector = YOLO('D:/license_plate-1/models/yolov8/license_plate_detector.pt')

# Mở video
cap = cv2.VideoCapture('C:/Users/gates/Downloads/Video/2103099-uhd_3840_2160_30fps.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Phát hiện biển số
    results = license_plate_detector(frame)[0]

    for detection in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Cắt biển số
        license_plate_crop = frame[y1:y2, x1:x2]
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

        # Nhận diện ký tự
        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

        # Hiển thị kết quả nếu có
        if license_plate_text is not None:
            print(f"Biển số: {license_plate_text} - Độ tin cậy: {license_plate_text_score:.2f}")
            # Vẽ bounding box lên frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Vẽ text lên frame
            cv2.putText(frame, license_plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Hiển thị frame
    cv2.imshow("License Plate Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
