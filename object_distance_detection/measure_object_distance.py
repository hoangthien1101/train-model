# http://www.pysource.com

from realsense_camera import WebcamCamera  # Đổi thành WebcamCamera
from object_detection import ObjectDetection
import cv2

# Create the Camera object (sử dụng webcam thay vì Realsense)
camera = WebcamCamera()

# Create the Object Detection object
object_detection = ObjectDetection()

while True:
    # Get frame from webcam camera (chỉ có color_image, không có depth_image)
    ret, color_image, _ = camera.get_frame_stream()  # Bỏ depth_image vì webcam không có dữ liệu độ sâu
    if not ret:
        break  # Nếu không lấy được khung hình, thoát vòng lặp

    height, width = color_image.shape[:2]

    # region vẽ tâm khung hình
    center_x, center_y = width // 2, height // 2
    
    # Hàm `draw_object_info` của bạn hiện đang nhận cả color_image và depth_image
    # Bạn có thể cần sửa đổi để nó hoạt động chỉ với color_image nếu không có depth_image
    object_detection.draw_object_info(color_image)  # Đưa None thay cho depth_image nếu cần

    # Vẽ trục ngang: từ (0, center_y) đến (width, center_y)
    color_image = cv2.line(color_image, (0, center_y), (width, center_y), (0, 255, 0), 2)

    # Vẽ trục dọc: từ (center_x, 0) đến (center_x, height)
    color_image = cv2.line(color_image, (center_x, 0), (center_x, height), (0, 255, 0), 2)
    # endregion

    # Hiện vị trí hiện tại của tâm bảng
    object_detection.Xac_dinh_vi_tri(color_image)

    # Hiển thị hình ảnh màu
    cv2.imshow("Color Image", color_image)
    
    # Nhấn phím Esc để thoát
    key = cv2.waitKey(1)
    if key == 27:  # Phím Esc
        break

# Giải phóng camera
camera.release()
cv2.destroyAllWindows()
