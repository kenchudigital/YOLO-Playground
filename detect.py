from ultralytics import YOLO
import cv2

TESTING_IMAGE_PATH = 'test_image.jpg'
NAME = 'elderly_fall_detection_cpu'
model = YOLO(f'runs/detect/{NAME}/weights/best.pt')
print(model.names) 

class_names = {
    0: "Fall Detected",
    1: "Walking",
    2: "Sitting"
}

image = cv2.imread(TESTING_IMAGE_PATH)
results = model(image)

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Get class ID and confidence
        class_id = int(box.cls)
        confidence = float(box.conf)
        
        # Draw bounding box
        color = (0, 0, 255) if class_id == 0 else (0, 255, 0)  # Red for falls
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = f"{class_names[class_id]}: {confidence:.2f}"
        cv2.putText(image, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

cv2.imwrite('detected_output.jpg', image)
cv2.imshow('Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()