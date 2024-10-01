# #http://www.pysource.com
# import numpy as np
# from ultralytics import YOLO
# import random
# import colorsys
# import torch
# import cv2
# from realsense_camera import WebcamCamera

# # Set random seed
# random.seed(2)


# class ObjectDetection:
#     #thay doi duong dan cua model phu hop voi thiet bi
#     def __init__(self, weights_path="D:\\WorkSpace\\Robocon\\train model\\train_model\\runs\\detect\\train2\\weights\\best.pt"): 
#         # Load Network
#         self.weights_path = weights_path
#         self.colors = self.random_colors(800)

#         # Load Yolo
#         self.model = YOLO(self.weights_path)
#         self.classes = self.model.names

#         # Load Default device
#         if torch.backends.mps.is_available():
#             self.device = torch.device("mps")
#         elif torch.cuda.is_available():
#             self.device = torch.device(0)
#         else:
#             self.device = torch.device("cpu")

#     def get_id_by_class_name(self, class_name):
#         for i, name in enumerate(self.classes.values()):
#             if name.lower() == class_name.lower():
#                 return i
#         return -1

#     def random_colors(self, N, bright=False):
#         brightness = 255 if bright else 180
#         hsv = [(i / N + 1, 1, brightness) for i in range(N + 1)]
#         colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
#         random.shuffle(colors)
#         return colors

#     def detect(self, frame, imgsz=1280, conf=0.25, nms=True, classes=None, device=None):
#         # Filter classes
#         filter_classes = classes if classes else None
#         device = device if device else self.device
#         # Detect objects
#         results = self.model.predict(source=frame, save=False, save_txt=False,
#                                      imgsz=imgsz,
#                                      conf=conf,
#                                      nms=nms,
#                                      classes=filter_classes,
#                                      half=False,
#                                      device=device)

#         # Get the first result from the array as we are only using one image
#         result = results[0]
#         # Get bboxes
#         bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
#         # Get class ids
#         class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
#         # Get scores
#         scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
#         return bboxes, class_ids, scores

#     def draw_object_info(self, color_image):
#         # Get the object detection
#         bboxes, class_ids, scores = self.detect(color_image)
#         for bbox, class_id, score in zip(bboxes, class_ids, scores):
#             x, y, x2, y2 = bbox
#             color = self.colors[class_id]
#             cv2.rectangle(color_image, (x, y), (x2, y2), color, 2)

#             # Display name
#             class_name = self.classes[class_id]
#             cv2.putText(color_image, f"{class_name}", (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

#             # Optionally calculate distance if you have that functionality
#             # distance = your_camera.get_distance_point(color_image, cx, cy)
#             # Draw distance if you have that functionality

#     def Xac_dinh_vi_tri(self, color_image):
#         height, width = color_image.shape[:2]
#         height, width = height // 2, width // 2
     
import numpy as np
from ultralytics import YOLO
import random
import colorsys
import torch
import cv2
from realsense_camera import WebcamCamera

class ObjectDetection:
    # thay doi duong dan cua model phu hop voi thiet bi
    def __init__(self, weights_path="D:\\WorkSpace\\Robocon\\train model\\train_model\\runs\\detect\\train2\\weights\\best.pt"): 
        # Load Network
        self.weights_path = weights_path
        self.colors = self.random_colors(800)

        # Load Yolo
        self.model = YOLO(self.weights_path)
        self.classes = self.model.names

        # Load Default device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device(0)
        else:
            self.device = torch.device("cpu")

    def random_colors(self, N, bright=False):
        brightness = 255 if bright else 180
        hsv = [(i / N + 1, 1, brightness) for i in range(N + 1)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def detect(self, frame, imgsz=1280, conf=0.25, nms=True, classes=None, device=None):
        # Filter classes
        filter_classes = classes if classes else None
        device = device if device else self.device
        # Detect objects
        results = self.model.predict(source=frame, save=False, save_txt=False,
                                     imgsz=imgsz,
                                     conf=conf,
                                     nms=nms,
                                     classes=filter_classes,
                                     half=False,
                                     device=device)

        # Get the first result from the array as we are only using one image
        result = results[0]
        # Get bboxes
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        # Get class ids
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        # Get scores
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
        return bboxes, class_ids, scores

    def draw_object_info(self, color_image):
        # Get the object detection
        bboxes, class_ids, scores = self.detect(color_image)
        for bbox, class_id, score in zip(bboxes, class_ids, scores):
            x, y, x2, y2 = bbox
            color = self.colors[class_id]
            cv2.rectangle(color_image, (x, y), (x2, y2), color, 2)

            # Display name
            class_name = self.classes[class_id]
            cv2.putText(color_image, f"{class_name}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    def Xac_dinh_vi_tri(self, color_image):
        # Lấy kích thước của khung hình
        height, width = color_image.shape[:2]
        center_x = width // 2  # Xác định tọa độ x của tâm

        # Lấy tọa độ của đối tượng sau khi detect
        bboxes, class_ids, scores = self.detect(color_image)

        for bbox, class_id, score in zip(bboxes, class_ids, scores):
            x, y, x2, y2 = bbox
            object_x = (x + x2) // 2  # Lấy tọa độ x ở giữa đối tượng

            # Tính khoảng cách từ object_x đến center_x
            delta_x = object_x - center_x

            # Chuyển đổi delta_x sang phạm vi từ -255 đến 255
            normalized_x = int((delta_x / center_x) * 255)

            # In ra terminal tọa độ theo yêu cầu
            print(f"Tọa độ đối tượng (theo trục X): {normalized_x}")

            # Bạn có thể trả về giá trị normalized_x hoặc thực hiện thêm xử lý khác ở đây
            return normalized_x

                
                

