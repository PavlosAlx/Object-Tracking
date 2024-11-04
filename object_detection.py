from ultralytics import YOLO
import cv2

from information import class_colors, classNames, excluded_classes

# YOLO weights
model = YOLO("yolov8l.pt")

cap = cv2.VideoCapture(0)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Width: {width}, Height: {height}")

width = 3840
height = 2160


cap.set(3, width)  # Set width
cap.set(4, height)  # Set height

line_thickness = 2
line_intensity = 0.9
src_img_intensity = 1

previous_positions = {}
line_canvas = None

while True:
    success, img = cap.read()
    if not success:
        break
    
    if line_canvas is None:
        line_canvas = img.copy()
        line_canvas[:] = (0, 0, 0)  # Set canvas to black

    results = model.track(img, stream=True)

    # Loop through detections and track objects
    for r in results:
        boxes = r.boxes

        for box in boxes:

            cls = int(box.cls[0])
            class_name = classNames[cls]

            if class_name in excluded_classes:
                continue

            # Bounding box coordinates,  bounding box, confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            confidence = round(float(box.conf[0]), 2)
            
            object_id = cls

            # Calculate the current position (center of the bounding box)
            current_position = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)

            # Get the color for this class from the class_colors dictionary, or default to white if not found
            color = class_colors.get(class_name, (255, 255, 255))
            
            if object_id in previous_positions:
                previous_position = previous_positions[object_id]
                cv2.line(line_canvas, previous_position, current_position, color, line_thickness)  # Draw line on canvas

            # Update previous position
            previous_positions[object_id] = current_position

            # Annotate image with class name and confidence
            cv2.putText(img, f'{class_name} {confidence}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Overlay the line canvas onto the frame

    #img_with_lines = cv2.addWeighted(img, 0.7, line_canvas, 0.3, 0)
    #img_with_lines = cv2.addWeighted(img, 1 - line_intensity, line_canvas, line_intensity, 0) 
    img_with_lines = cv2.addWeighted(img, src_img_intensity, line_canvas, line_intensity, 0)
    
    cv2.imshow("Object Tracking with Persistent Lines", img_with_lines)

    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
