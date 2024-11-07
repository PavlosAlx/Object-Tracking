from ultralytics import YOLO
import time
import os
import sys
import cv2
import numpy as np
from information import class_colors, classNames, excluded_classes


def run_track():
    # Initialize YOLO
    model = YOLO("yolov8m.pt")

    # Configurable parameters
    line_thickness = 2
    line_intensity = 0.9
    src_img_intensity = 1
    resize_width = 640  # Resize image to 640x480 
    resize_height = 480

    cap = cv2.VideoCapture(0)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # Hardcoded if needed
    width = 1920
    height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print(f"Width: {width}, Height: {height}")

    previous_positions = {}  # Maps unique IDs to positions
    line_canvas = None

    start_time = time.time()

    while True:

        elapsed_time = time.time() - start_time
        if elapsed_time > 900:  # 900 = 15 minutes
            print("15 minutes have passed. Restarting the program.")
            os.execl(sys.executable, sys.executable, *sys.argv)  # Restart the script

        success, img = cap.read()
        if not success:
            break

        resized_img = cv2.resize(img, (resize_width, resize_height))

        # Create black canvas for drawing lines
        if line_canvas is None:
            line_canvas = img.copy()
            line_canvas[:] = (0, 0, 0)

        # Get YOLO detections
        results = model.track(resized_img, stream=True, persist = True)
        
        # Process each detected object
        for r in results:
            for box in r.boxes:

                cls = int(box.cls[0])
                class_name = classNames[cls]

                if class_name in excluded_classes:
                    continue

                # bbox coordinates & current pos
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Scale coordinates back to the original resolution if necessary
                scale_x = width / resize_width
                scale_y = height / resize_height
                x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                current_position = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
                # track id & color
                track_id = int(box.id[0]) 
                color = class_colors.get(class_name, (255, 255, 255))

                # # skip tracking "person"
                # if class_name in excluded_classes:
                #     # Annotate with class and confidence
                #     confidence = round(float(box.conf[0]), 2)
                #     cv2.putText(img, f'{class_name} {confidence}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                #     continue  

                # If the object has previous positions, use them to draw a smoother curve
                if track_id in previous_positions:
                    previous_positions[track_id].append(current_position)

                    # Limit history to 10 points to avoid overcrowded paths
                    # if len(previous_positions[track_id]) > 100:
                    #     previous_positions[track_id].pop(0)

                    # smoother curves than cv2.line
                    points = np.array(previous_positions[track_id], np.int32)
                    cv2.polylines(line_canvas, [points], isClosed=False, color=color, thickness=line_thickness)
                
                else:
                    # Initialize with the first position
                    previous_positions[track_id] = [current_position]

                # Annotate with class and confidence
                confidence = round(float(box.conf[0]), 2)
                cv2.putText(img, f'{class_name} {confidence}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Overlay the line canvas onto the frame
        img_with_lines = cv2.addWeighted(img, src_img_intensity, line_canvas, line_intensity, 0)

        cv2.imshow("Object Tracking with Persistent Curved Lines", img_with_lines)
        
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    run_track() 