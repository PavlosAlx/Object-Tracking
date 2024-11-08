import time
import os
import sys
import cv2
import numpy as np
import websocket
import base64
import json
from information import class_colors, classNames, excluded_classes

# Global dictionary to store previous positions for tracking
previous_positions = {}  # Maps unique IDs to positions
line_canvas = None  # Canvas for drawing paths

# Configurable parameters for drawing
line_thickness = 2
line_intensity = 0.9
src_img_intensity = 1
resize_width = 640  # Resize image to 640x480 
resize_height = 480

# Video capture setup
cap = cv2.VideoCapture(0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# Hardcoded if needed
width = 1920
height = 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

print(f"Width: {width}, Height: {height}")

def on_message(ws, message):
    """
    This function handles incoming WebSocket messages (the inference result).
    """
    global previous_positions, line_canvas
    
    data = json.loads(message)
    img = data['image']  # The image with bounding boxes drawn (returned by Azure model)
    results = data['predictions']  # Detection results (bounding boxes, class IDs, and confidences)

    # Create black canvas for drawing lines if not created
    if line_canvas is None:
        line_canvas = img.copy()
        line_canvas[:] = (0, 0, 0)

    # Process each detected object
    for detection in results:
        cls = detection['class']
        class_name = classNames.get(cls, "Unknown")
        
        if class_name in excluded_classes:
            continue

        # Bounding box coordinates
        x1, y1, x2, y2 = map(int, detection['bbox'])

        # Scale coordinates back to the original resolution if necessary
        scale_x = width / resize_width
        scale_y = height / resize_height
        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        current_position = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
        track_id = detection['id']  # Track ID
        color = class_colors.get(class_name, (255, 255, 255))  # Get color for the track

        # If the object has previous positions, update the track
        if track_id in previous_positions:
            previous_positions[track_id].append(current_position)

            # Limit the number of points to avoid overcrowding paths
            if len(previous_positions[track_id]) > 100:
                previous_positions[track_id].pop(0)

            # Draw the path for this object
            points = np.array(previous_positions[track_id], np.int32)
            cv2.polylines(line_canvas, [points], isClosed=False, color=color, thickness=line_thickness)
        
        else:
            # Initialize the track with the first position
            previous_positions[track_id] = [current_position]

        # Annotate with class name and confidence
        confidence = round(detection['confidence'], 2)
        cv2.putText(img, f'{class_name} {confidence}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Overlay the path (line_canvas) onto the original image
    img_with_lines = cv2.addWeighted(img, src_img_intensity, line_canvas, line_intensity, 0)
    cv2.imshow("Object Tracking with Persistent Curved Lines", img_with_lines)

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("### closed ###")

def on_open(ws):
    print("WebSocket connection opened")

def run_track():
    # WebSocket Server URL (replace with your actual WebSocket endpoint URL)
    websocket_url = "ws://<your-azure-websocket-endpoint>"
    
    # Start the WebSocket listener in a separate thread
    ws = websocket.WebSocketApp(websocket_url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    
    # Create the WebSocket connection
    ws.run_forever()

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

        # Convert the frame to base64 encoding for transmission
        _, buffer = cv2.imencode('.jpg', resized_img)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        # Prepare the payload to send to Azure ML via WebSocket
        data = {
            'image': jpg_as_text
        }

        # Send the frame to the Azure WebSocket server
        try:
            ws.send(json.dumps(data))
        except Exception as e:
            print(f"Error sending data to WebSocket: {e}")

        # Wait for the response and process it in the on_message callback
        cv2.waitKey(1)  # Process frames at a normal speed

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    run_track()
