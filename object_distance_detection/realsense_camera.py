import cv2
import numpy as np

class WebcamCamera:
    def __init__(self):
        print("Loading webcam")
        # Khởi tạo video capture với webcam (0 là id mặc định của webcam đầu tiên)
        self.cap = cv2.VideoCapture(0)
        
        # Đặt độ phân giải cho webcam
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            exit(1)

    def get_frame_stream(self):
        # Đọc khung hình từ webcam
        ret, frame = self.cap.read()
        
        if not ret:
            print("Error: Could not read frame from webcam.")
            return False, None, None
        
        # Chuyển đổi sang hình ảnh màu nếu cần
        color_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return True, color_image, None

    def release(self):
        # Giải phóng tài nguyên webcam
        self.cap.release()
   


