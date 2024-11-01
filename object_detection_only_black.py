from ultralytics import YOLO
import cv2

from information import class_colors, classNames

# YOLO weights
model = YOLO("yolov8l.pt")

# Initialize webcam capture
cap = cv2.VideoCapture(0)
cap.set(3, 1180)  # Set width
cap.set(4, 900) # Set height

thickness = 2

# Initialize tracking variables
previous_positions = {}
line_canvas = None

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Initialize line canvas to black if it's the first frame
    if line_canvas is None:
        height, width, _ = img.shape
        line_canvas = img.copy()
        line_canvas[:] = (0, 0, 0)  # Black background

    results = model.track(img, stream=True)

    # Loop through detections and track objects
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Get confidence score and class index
            confidence = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])
            class_name = classNames[cls]

            # Object ID based on class
            object_id = cls

            # Calculate the current position (center of the bounding box)
            current_position = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)

            # Get the color for this class from the class_colors dictionary
            color = class_colors.get(class_name, (255, 255, 255))
            
            # Draw line on black canvas if previous position exists
            if object_id in previous_positions:
                previous_position = previous_positions[object_id]
                cv2.line(line_canvas, previous_position, current_position, color, thickness)

            # Update previous position
            previous_positions[object_id] = current_position

    # Display only the line canvas
    cv2.imshow("Object Movement on Black Background", line_canvas)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
