from ultralytics import YOLO
import cv2
import numpy as np
from information import class_colors, classNames, excluded_classes

# Initialize YOLO
model = YOLO("yolov8m.pt")

# Configurable parameters
line_thickness = 2
line_intensity = 0.9
src_img_intensity = 1

cap = cv2.VideoCapture(0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Width: {width}, Height: {height}")
# width = 1000
# height = 800
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

previous_positions = {}  # Maps unique IDs to positions
line_canvas = None

# black canvas for output
output_canvas = np.zeros((int(height), int(width), 3), dtype=np.uint8)


while True:
    success, img = cap.read()
    if not success:
        break
    
    # Create black canvas for drawing lines
    if line_canvas is None:
        line_canvas = img.copy()
        line_canvas[:] = (0, 0, 0)

    # Get YOLO detections
    results = model.track(img, stream=True, persist=True)
    
    # Process each detected object
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            class_name = classNames[cls]

            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(output_canvas, (x1, y1), (x2, y2), (255, 0, 255), 2)

            # Calculate the current position (center of the bounding box)
            current_position = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)

            # Track id & color
            track_id = int(box.id[0])  # Use the 'id' attribute instead of 'track_id'
            color = class_colors.get(class_name, (255, 255, 255))

            # Check if we need to skip tracking for this class
            if class_name in excluded_classes:
                # Annotate with class and confidence
                confidence = round(float(box.conf[0]), 2)
                cv2.putText(output_canvas, f'{class_name} {confidence}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                continue  # Skip tracking logic for this class

            # If the object has previous positions, use them to draw a smoother curve
            if track_id in previous_positions:
                previous_positions[track_id].append(current_position)

                # Limit history to 10 points to avoid overcrowded paths
                # if len(previous_positions[track_id]) > 10:
                #     previous_positions[track_id].pop(0)

                # smoother curves
                points = np.array(previous_positions[track_id], np.int32)
                cv2.polylines(line_canvas, [points], isClosed=False, color=color, thickness=line_thickness)
            
            else:
                # Initialize with the first position
                previous_positions[track_id] = [current_position]

            # Annotate with class and confidence
            confidence = round(float(box.conf[0]), 2)
            cv2.putText(output_canvas, f'{class_name} {confidence}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Overlay the line canvas onto the black background
    output_canvas = cv2.addWeighted(output_canvas, src_img_intensity, line_canvas, line_intensity, 0)

    # Display the result on a black background
    cv2.imshow("Object Tracking on Black Background", output_canvas)

    # Reset the output_canvas for the next frame
    output_canvas.fill(0)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
