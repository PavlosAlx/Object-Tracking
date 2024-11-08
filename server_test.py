# server.py

from flask import Flask
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import base64
import cv2
import numpy as np
from information import class_colors, classNames, excluded_classes

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize YOLO
model = YOLO("yolov8l.pt")

# Configurable parameters
line_thickness = 2
line_intensity = 0.9
src_img_intensity = 1
resize_width = 640  # Resize image to 640x480
resize_height = 480

previous_positions = {}  # Maps unique IDs to positions
line_canvas = None

# Define your class names, colors, and excluded classes
class_names = model.names  # Assuming model.names contains class names
excluded_classes = ['person']  # Example: Exclude 'person' class
class_colors = {name: [int(c) for c in np.random.choice(range(256), size=3)] for name in class_names.values()}

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('video_frame')
def handle_video_frame(data):
    global line_canvas, previous_positions
    # Decode the image
    img_data = base64.b64decode(data)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        print("Received empty frame")
        return

    # Resize image
    resized_img = cv2.resize(img, (resize_width, resize_height))

    # Initialize line canvas
    if line_canvas is None or line_canvas.shape != resized_img.shape:
        line_canvas = np.zeros_like(resized_img)

    # Run YOLO inference
    results = model.track(resized_img, stream=True, persist=True)
    
    # Process detections
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            class_name = class_names.get(cls, 'Unknown')

            if class_name in excluded_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            current_position = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
            track_id = int(box.id[0])
            color = class_colors.get(class_name, (255, 255, 255))

            # Draw tracking lines
            if track_id in previous_positions:
                previous_positions[track_id].append(current_position)
                if len(previous_positions[track_id]) > 100:
                    previous_positions[track_id].pop(0)
                points = np.array(previous_positions[track_id], np.int32)
                cv2.polylines(line_canvas, [points], isClosed=False, color=color, thickness=line_thickness)
            else:
                previous_positions[track_id] = [current_position]

            # Draw bounding box and label
            confidence = round(float(box.conf[0]), 2)
            cv2.rectangle(resized_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(resized_img, f'{class_name} {confidence}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Overlay lines onto the image
    img_with_lines = cv2.addWeighted(resized_img, src_img_intensity, line_canvas, line_intensity, 0)

    # Encode image as JPEG
    # print(img_with_lines.shape)
    _, img_encoded = cv2.imencode('.jpg', img_with_lines)
    img_bytes = img_encoded.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    # # img_with_lines = img_with_lines.tobytes()
    # # img_base64 = base64.b64encode(img_with_lines).decode('utf-8')

    # Send processed frame back to client
    emit('processed_frame', img_base64)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8888)