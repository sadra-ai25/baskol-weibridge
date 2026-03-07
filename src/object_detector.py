from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        print(f"مدل YOLO از مسیر {model_path} با موفقیت بارگذاری شد.")

    def detect(self, frame, confidence_threshold=0.5):
        results = self.model(frame, conf=confidence_threshold, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                detections.append({
                    "class": class_name,
                    "confidence": conf,
                    "box": [x1, y1, x2, y2]
                })
        return detections