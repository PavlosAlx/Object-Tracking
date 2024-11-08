import cv2
import base64
import time
import threading
import socketio
import sys
import numpy as np

width, height = 640, 480  # Reduced dimensions for transmission
original_width, original_height = 1920, 1080  # Original dimensions

# Create a Socket.IO client
sio = socketio.Client()

# Initialize video capture
cap = cv2.VideoCapture(0)

# Function to send video frames to the server
def send_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to smaller dimensions before sending
        frame = cv2.resize(frame, (width, height))

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')

        # Send the frame to the server
        sio.emit('video_frame', frame_base64)

        # Add a small delay to control frame rate
        time.sleep(0.1)

@sio.event
def connect():
    print('Connected to server')

@sio.event
def disconnect():
    print('Disconnected from server')
    cap.release()
    cv2.destroyAllWindows()
    # Instead of sys.exit(), we keep the client running for reconnects
    sio.connect('http://45.77.227.214:8888')  # Reconnect if needed

@sio.on('processed_frame')
def on_processed_frame(data):
    # Decode the base64 image
    img_bytes = base64.b64decode(data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is not None:
        # Resize frame back to original dimensions
        img = cv2.resize(img, (original_width, original_height))
        cv2.imshow('Processed Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sio.disconnect()
    else:
        print("Failed to decode image")


if __name__ == '__main__':
    # Connect to the server
    sio.connect('http://45.77.227.214:8888')

    # Start sending frames in a separate thread
    thread = threading.Thread(target=send_frames)
    thread.start()

    # Keep the main thread alive to receive events
    sio.wait()
