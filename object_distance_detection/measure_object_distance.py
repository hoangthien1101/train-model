from realsense_camera import WebcamCamera  # Đổi thành WebcamCamera
from object_detection import ObjectDetection
import cv2

# Tạo đối tượng Camera (sử dụng webcam)
camera = WebcamCamera()

# Tạo đối tượng phát hiện đối tượng
object_detection = ObjectDetection()

while True:
    # Lấy khung hình từ webcam (chỉ có color_image, không có depth_image)
    ret, color_image, _ = camera.get_frame_stream()  # Sửa để chỉ trả về color_image từ webcam
    if not ret:
        break  # Nếu không lấy được khung hình, thoát vòng lặp

    height, width = color_image.shape[:2]

    # region Vẽ tâm khung hình
    center_x, center_y = width // 2, height // 2
    
    # Vẽ trục ngang: từ (0, center_y) đến (width, center_y)
    color_image = cv2.line(color_image, (0, center_y), (width, center_y), (0, 255, 0), 2)

    # Vẽ trục dọc: từ (center_x, 0) đến (center_x, height)
    color_image = cv2.line(color_image, (center_x, 0), (center_x, height), (0, 255, 0), 2)
    # endregion

    # Phát hiện đối tượng và vẽ thông tin lên hình ảnh
    bboxes, class_ids, scores = object_detection.detect(color_image)

    # Vòng lặp qua các đối tượng được phát hiện
    for bbox, class_id in zip(bboxes, class_ids):
        x, y, x2, y2 = bbox
        object_center_x = (x + x2) // 2
        object_center_y = (y + y2) // 2

        # Hiển thị tọa độ của đối tượng trên khung hình
        relative_x = int((object_center_x - center_x) * (255 / center_x))  # Từ -255 đến 255
        relative_y = int((center_y - object_center_y) * (255 / center_y))  # Từ -255 đến 255

        # In tọa độ ra terminal
        print(f"Toa do: ({relative_x}, {relative_y})")

        # Hiển thị tọa độ vị trí lên hình ảnh
        cv2.putText(color_image, f"Toa do: ({relative_x}, {relative_y})", 
                    (object_center_x, object_center_y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(color_image, f"Backboard",
                    (object_center_x, object_center_y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        # Vẽ hình chữ nhật quanh đối tượng
        cv2.rectangle(color_image, (x, y), (x2, y2), (255, 0, 0), 2)

    # Hiển thị khung hình màu
    cv2.imshow("Color Image", color_image)          

    # Nhấn phím Esc để thoát
    key = cv2.waitKey(1)
    if key == 27:  # Phím Esc
        break

# Giải phóng camera
camera.release()
cv2.destroyAllWindows()
