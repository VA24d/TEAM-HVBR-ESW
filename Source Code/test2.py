#best working model, with distance averaged and walls detected 
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os
import time
from datetime import datetime
import serial
import threading
from queue import Queue
from collections import deque
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import pyttsx3

class DistanceFilter:
    def __init__(self, buffer_size=3, threshold=2.0):
        self.buffer = []
        self.buffer_size = buffer_size
        self.threshold = threshold

    def get_average(self):
        if not self.buffer:
            return None
        return sum(self.buffer) / len(self.buffer)

    def should_update(self, new_value):
        if new_value is None:
            return False
        
        current_avg = self.get_average()
        if current_avg is None:
            return True
            
        return abs(new_value - current_avg) > self.threshold

    def update(self, new_value):
        if new_value is None:
            return self.get_average()

        if not self.buffer or self.should_update(new_value):
            self.buffer.append(new_value)
            if len(self.buffer) > self.buffer_size:
                self.buffer.pop(0)
        
        return self.get_average()

def try_open_serial(port='COM6', max_attempts=3, wait_time=2):
    """Attempt to open serial port with retries and proper cleanup."""
    import serial.tools.list_ports
    
    # First, check if port exists
    available_ports = [p.device for p in serial.tools.list_ports.comports()]
    if port not in available_ports:
        print(f"Port {port} not found. Available ports: {available_ports}")
        return None

    # Close any existing serial connections on this port
    if 'ser' in globals():
        try:
            globals()['ser'].close()
        except:
            pass
    
    for attempt in range(max_attempts):
        try:
            # Force close any existing connections
            try:
                temp_ser = serial.Serial(port)
                temp_ser.close()
            except:
                pass
            
            # Wait for port to be released
            time.sleep(wait_time)
            
            # Try to open new connection
            ser = serial.Serial(
                port=port,
                baudrate=115200,
                timeout=1,
                write_timeout=1
            )
            print(f"Successfully connected to {port}")
            return ser
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_attempts - 1:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
    
    print(f"Failed to open {port} after {max_attempts} attempts")
    return None

# Initialize serial connection
try:
    ser = try_open_serial()
except Exception as e:
    print(f"Serial initialization error: {e}")
    ser = None

# Add configuration parameters
FRAME_WIDTH = 640  # Reduced frame size
FRAME_HEIGHT = 480
PROCESS_EVERY_N_FRAMES = 1  # Process every nth frame
QUEUE_SIZE = 5  # Frame queue size
BATCH_SIZE = 1  # YOLO batch size
DISTANCE_THRESHOLD = 50  # Configurable distance threshold in cm

debug = True
path_message = ""

# Add at the start of the program
if 'ser' in globals() and ser is not None:
    ser.close()

prev_line = None   # Variable to store the previous line read from serial

distance_filter = DistanceFilter(buffer_size=3, threshold=2.0)

# Function to read distance from ESP32
def read_distance():
    global prev_line, distance_filter
    if ser is None:
        return None
    try:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8').strip()
            if not line:
                return distance_filter.get_average()
            try:
                raw_distance = float(line)
                filtered_distance = distance_filter.update(raw_distance)
                prev_line = filtered_distance
                if debug:
                    print(f"Raw: {raw_distance:.2f} cm, Filtered: {filtered_distance:.2f} cm")  # Debug print
                return filtered_distance
            except ValueError:
                return distance_filter.get_average()
    except Exception as e:
        print(f"Error reading serial data: {e}")
        return distance_filter.get_average()
    return distance_filter.get_average()

# Initialize speech queue and engine
speech_queue = Queue()

last_spoken = {"text": "", "time": 0}  # Change to dictionary to track timing

def speech_worker():
    global last_spoken
    engine = pyttsx3.init()
    while True:
        text = speech_queue.get()
        if text is None:
            break
        
        # Allow same message after 1 second
        current_time = time.time()
        if text == last_spoken["text"] and (current_time - last_spoken["time"]) < 1:
            continue
            
        engine.say(text)
        engine.runAndWait()
        last_spoken = {"text": text, "time": current_time}

# Start the speech thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

def speak_async(text: str):
    """
    Convert text to speech asynchronously, allowing the program to continue execution.

    Parameters:
    text (str): The text to be spoken.
    """
    speech_queue.put(text)

# Specify the device here: 'cpu' or 'gpu'.
# Leave it as an empty string ('') to prompt the user.
specified_device = "gpu"  # Change this to 'cpu', 'gpu', or leave as ''.

# Specify the desired output FPS for video processing.
desired_fps = 30  # Change this value to adjust the desired FPS for the output video.

# Determine the device to use based on the specified_device variable.
if specified_device.lower() == "gpu" and torch.cuda.is_available():
    device = "cuda"
    print("CUDA is available. Using GPU for processing.")
elif specified_device.lower() == "cpu":
    device = "cpu"
    print("Using CPU for processing as specified.")
else:
    # If the specified device is invalid or empty, prompt the user.
    user_device = input("Enter 'gpu' to use GPU (if available) or 'cpu' to use the processor: ").strip().lower()
    if (user_device == "gpu" and torch.cuda.is_available()):
        device = "cuda"
        print("CUDA is available. Using GPU for processing.")
    else:
        device = "cpu"
        if user_device == "gpu":
            print("GPU was requested but is not available. Using CPU instead.")
        else:
            print("Using CPU for processing as requested.")

print(f"Using device: {device}")

# Optimize torch for CPU multi-threading if using CPU
if device == "cpu":
    # Set the number of threads PyTorch can use
    torch.set_num_threads(16)  # Adjust this based on your CPU's core count
    torch.set_num_interop_threads(16)  # Adjust this as well

    # Print the thread configuration for debugging
    print(f"PyTorch is set to use {torch.get_num_threads()} threads.")
    print(f"PyTorch interop threads set to {torch.get_num_interop_threads()}.")

def load_model(model_path):
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found.")
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    try:
        model = YOLO(model_path)
        model.to(device)
        print("YOLOv8 model loaded successfully.")
        return model
    except Exception as e:
        print(f"Failed to load the model: {e}")
        raise e

try:
    model = load_model('yolo11x-seg.pt')  # Use the detection model
except Exception as e:
    print("Exiting due to model loading failure.")
    print(e)
    exit(1)

# Retrieve the list of class names from the model
class_names = model.names  # Dictionary mapping class IDs to class names

def resize_frame(frame):
    return cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

class FrameProcessor:
    def __init__(self):
        self.frame_queue = Queue(maxsize=2)  # Reduced queue size
        self.result_queue = Queue(maxsize=2)
        self.last_results = None
        self.processing = True
        self.frame_info = {}  # Store frame information
        self.current_distance = None  # Store current distance
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.start()

    def _process_frames(self):
        while self.processing:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                self.current_distance = read_distance()  # Update current distance
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame with optimized settings
                results = model(rgb_frame, 
                             conf=0.3,
                             iou=0.45,
                             verbose=False,
                             device=device)
                
                self.last_results = results
                processed_frame = process_frame(frame, self)  # Pass processor instance
                self.result_queue.put(processed_frame)

    def stop(self):
        self.processing = False
        self.processing_thread.join()

def process_frame(frame, processor):  # Add processor parameter
    try:
        # Initialize all flags and variables at the start
        center_clear = True  # Default to True
        left_clear = True
        right_clear = True
        wall_detected = False
        wall_message = ""
        path_detected = False
        sticker_message = ''
        stickers_detected = False
        path_message = ''
        rounded_distance = None

        # Get current distance
        distance = processor.current_distance
        if distance is not None:
            rounded_distance = round(distance)
            distance_msg = f"Distance: {rounded_distance} cm"
            cv2.putText(frame, distance_msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            
            if rounded_distance < DISTANCE_THRESHOLD:
                warning_msg = "WARNING: Obstacle very close!"
                cv2.putText(frame, warning_msg, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 0, 255), 2)

        # Convert frame to RGB and get results
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb_frame, conf=0.25, iou=0.45, verbose=False)
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]

        # First check for wall condition
        no_objects_detected = len(results[0].boxes) == 0
        if distance is not None and distance < DISTANCE_THRESHOLD:
            if no_objects_detected:
                wall_detected = True
                wall_message = f"Wall ahead at {rounded_distance} cm"
                cv2.putText(frame, wall_message, (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 0, 255), 2)
                center_clear = False  # Update center_clear flag

        # Rest of your existing process_frame code
        # ...existing code...

        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]  # Height, width

        # Initialize clear flags
        center_clear = True
        left_clear = True
        right_clear = True

        # Divide frame into regions
        left_boundary = frame_width // 3
        right_boundary = 2 * frame_width // 3

        # Draw dividing lines on the frame
        cv2.line(frame, (left_boundary, 0), (left_boundary, frame_height), (255, 0, 0), 2)
        cv2.line(frame, (right_boundary, 0), (right_boundary, frame_height), (255, 0, 0), 2)

        # Create a mask to exclude detected persons from color detection
        exclusion_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

        # Initialize variables to accumulate occupied area in each region
        left_occupied_rows = np.zeros(frame_height, dtype=np.uint8)
        center_occupied_rows = np.zeros(frame_height, dtype=np.uint8)
        right_occupied_rows = np.zeros(frame_height, dtype=np.uint8)

        # Create an obstacle mask for path detection
        obstacle_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

        # List to store detected colored objects for visual clues
        colored_objects = []  # Each element: (x1, y1, x2, y2, color)

        # List to store detected colored objects in grid area for direction detection
        grid_colored_objects = []

        # Define the grid area in the bottom of the center region
        offset = frame_width // 8
        height_offset = frame_height // 4

        grid_points = np.array([
            [left_boundary + offset, frame_height - height_offset],  # Top left
            [right_boundary - offset, frame_height - height_offset],  # Top right
            [right_boundary, frame_height],  # Bottom right
            [left_boundary, frame_height]   # Bottom left
        ], dtype=np.int32)

        # Draw the pink grid (parallelogram)
        cv2.polylines(frame, [grid_points], isClosed=True, color=(255, 0, 255), thickness=2)

        # Create a mask for the grid area
        grid_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        cv2.fillPoly(grid_mask, [grid_points], color=255)

        # Process YOLO detections
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    class_name = class_names.get(cls_id, 'Unknown')

                    # Exclude persons from color detection
                    if class_name == 'person':
                        cv2.rectangle(exclusion_mask, (x1, y1), (x2, y2), 255, -1)  # Fill with white to exclude

                    # Draw bounding box around detected object with class name
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Put class label
                    label = f"{class_name}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)

                    # Draw the obstacle area on the obstacle mask
                    cv2.rectangle(obstacle_mask, (x1, y1), (x2, y2), color=255, thickness=-1)

                    # For each region, compute the overlapping x-range
                    # Left region
                    overlap_left_x1 = max(x1, 0)
                    overlap_left_x2 = min(x2, left_boundary)
                    if overlap_left_x1 < overlap_left_x2:
                        # Overlaps with left region
                        y1_clipped = max(y1, 0)
                        y2_clipped = min(y2, frame_height)
                        left_occupied_rows[y1_clipped:y2_clipped] = 1

                    # Center region
                    overlap_center_x1 = max(x1, left_boundary)
                    overlap_center_x2 = min(x2, right_boundary)
                    if overlap_center_x1 < overlap_center_x2:
                        # Overlaps with center region
                        y1_clipped = max(y1, 0)
                        y2_clipped = min(y2, frame_height)
                        center_occupied_rows[y1_clipped:y2_clipped] = 1

                    # Right region
                    overlap_right_x1 = max(x1, right_boundary)
                    overlap_right_x2 = min(x2, frame_width)
                    if overlap_right_x1 < overlap_right_x2:
                        # Overlaps with right region
                        y1_clipped = max(y1, 0)
                        y2_clipped = min(y2, frame_height)
                        right_occupied_rows[y1_clipped:y2_clipped] = 1

                    # If the object is not a person, perform color detection
                    if class_name != 'person':
                        # Extract the ROI of the object
                        object_roi = frame[y1:y2, x1:x2]
                        # Apply color detection on the object_roi
                        hsv_roi = cv2.cvtColor(object_roi, cv2.COLOR_BGR2HSV)

                        # Red color mask
                        lower_red1 = np.array([0, 70, 50])
                        upper_red1 = np.array([10, 255, 255])
                        lower_red2 = np.array([170, 70, 50])
                        upper_red2 = np.array([180, 255, 255])
                        mask_red1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
                        mask_red2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
                        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

                        # Blue color mask
                        lower_blue = np.array([100, 150, 50])
                        upper_blue = np.array([140, 255, 255])
                        mask_blue = cv2.inRange(hsv_roi, lower_blue, upper_blue)

                        # Apply morphological operations to reduce noise
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
                        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
                        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
                        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)

                        # Count the number of red and blue pixels
                        red_pixels = cv2.countNonZero(mask_red)
                        blue_pixels = cv2.countNonZero(mask_blue)
                        total_pixels = (x2 - x1) * (y2 - y1)

                        # Determine dominant color
                        color = None
                        if red_pixels > blue_pixels and red_pixels > total_pixels * 0.3:
                            color = 'red'
                            # Draw bounding box in red
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        elif blue_pixels > red_pixels and blue_pixels > total_pixels * 0.3:
                            color = 'blue'
                            # Draw bounding box in blue
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                        if color:
                            # Calculate the center of the object
                            object_center_x = (x1 + x2) // 2
                            object_center_y = (y1 + y2) // 2

                            # Check if the object is within the grid area
                            if cv2.pointPolygonTest(grid_points, (object_center_x, object_center_y), False) >= 0:
                                grid_colored_objects.append((x1, y1, x2, y2, color))

                            # Add to colored_objects for visual clues in other regions if needed
                            colored_objects.append((x1, y1, x2, y2, color))

        # Compute the occupied percentage in each region
        left_occupied_percentage = np.sum(left_occupied_rows) / frame_height * 100
        center_occupied_percentage = np.sum(center_occupied_rows) / frame_height * 100
        right_occupied_percentage = np.sum(right_occupied_rows) / frame_height * 100

        # Threshold for occupied percentage
        threshold = 10  # percentage

        # Update clear flags based on occupied percentage
        left_clear = left_occupied_percentage < threshold
        center_clear = center_occupied_percentage < threshold
        right_clear = right_occupied_percentage < threshold

        # Compute the overlap between the grid area and the obstacle mask
        grid_obstacle_overlap = cv2.bitwise_and(grid_mask, obstacle_mask)

        # Check if there are any obstacles in the grid area
        if cv2.countNonZero(grid_obstacle_overlap) == 0:
            # Path is clear in the grid area
            path_detected = True
        else:
            path_detected = False

        # Initialize sticker message
        sticker_message = ''
        stickers_detected = False

        # Direction detection in the grid area
        if len(grid_colored_objects) == 2:
            # Sort the objects by x coordinate
            grid_colored_objects.sort(key=lambda obj: obj[0])  # Sort by x1 position
            obj1 = grid_colored_objects[0]
            obj2 = grid_colored_objects[1]
            color1 = obj1[4]
            color2 = obj2[4]

            # Check if objects are adjacent (you can adjust the threshold)
            x_distance = abs(((obj1[0] + obj1[2]) // 2) - ((obj2[0] + obj2[2]) // 2))
            if x_distance < (right_boundary - left_boundary):
                # Objects are adjacent
                stickers_detected = True
                if color1 == 'blue' and color2 == 'blue':
                    sticker_message = 'Exit to North'
                elif color1 == 'red' and color2 == 'red':
                    sticker_message = 'Exit to South'
                elif color1 == 'blue' and color2 == 'red':
                    sticker_message = 'Exit to East'
                elif color1 == 'red' and color2 == 'blue':
                    sticker_message = 'Exit to West'
            else:
                sticker_message = 'Stickers not adjacent'

        elif len(colored_objects) == 2:
            # Visual clue detection in other regions
            # Sort the objects by x coordinate
            colored_objects.sort(key=lambda obj: obj[0])  # Sort by x1 position
            obj1 = colored_objects[0]
            obj2 = colored_objects[1]
            color1 = obj1[4]
            color2 = obj2[4]

            # Check if objects are adjacent (you can adjust the threshold)
            x_distance = abs(((obj1[0] + obj1[2]) // 2) - ((obj2[0] + obj2[2]) // 2))
            if x_distance < (right_boundary - left_boundary):
                # Objects are adjacent
                stickers_detected = True
                if color1 == 'red' and color2 == 'red':
                    sticker_message = 'Dead end'
                elif color1 == 'red' and color2 == 'blue':
                    sticker_message = 'You can turn right'
                elif color1 == 'blue' and color2 == 'red':
                    sticker_message = 'You can turn left'
                elif color1 == 'blue' and color2 == 'blue':
                    sticker_message = 'You can turn in both directions'
            else:
                sticker_message = 'Stickers not adjacent'

        else:
            sticker_message = ''

        path_message = ''
        # Display messages on frame
        if stickers_detected and sticker_message:
            # Display the sticker message
            cv2.putText(frame, sticker_message, (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2)
            # Removed speak_async here
        elif path_detected and center_clear:
            # Display "Path detected" message when both center and grid are clear
            path_message = 'Path ahead is clear'
            cv2.putText(frame, path_message, (10, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2)
            # Removed speak_async here
        else:
            # Determine path message based on clear flags
            path_message = ''
            if center_clear:
                path_message = 'Path ahead is clear'
            else:
                if left_clear and right_clear:
                    path_message = 'Left and right are free'
                elif left_clear:
                    path_message = 'Left path is clear'
                elif right_clear:
                    path_message = 'Right path is clear'
                else:
                    path_message = 'Stop'
            
            # Display path message on frame
            cv2.putText(frame, path_message, (10, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2)
            # Removed speak_async here

        # Update the processor's frame_info with rounded distance
        processor.frame_info = {
            'path_message': path_message,
            'sticker_message': sticker_message if stickers_detected else None,
            'distance': rounded_distance if rounded_distance is not None else None,
            'wall_message': wall_message if wall_detected else None
        }

    except Exception as e:
        print(f"Error in processing frame: {e}")

    return frame

# Add this new function after the process_frame function
def generate_scene_description(frame, results):
    frame_height, frame_width = frame.shape[:2]
    left_boundary = frame_width // 3
    right_boundary = 2 * frame_width // 3
    
    # Initialize lists for each section
    left_objects = []
    center_objects = []
    right_objects = []
    
    # Process YOLO detections
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                class_name = class_names.get(cls_id, 'Unknown')
                
                # Calculate center of object
                center_x = (x1 + x2) // 2
                
                # Determine which section the object is in
                if center_x < left_boundary:
                    left_objects.append(class_name)
                elif center_x < right_boundary:
                    center_objects.append(class_name)
                else:
                    right_objects.append(class_name)
    
    # Generate description
    description = "Scene Description:\n"
    
    if left_objects:
        description += "Left section: " + ", ".join(left_objects) + "\n"
    else:
        description += "Left section: Clear\n"
        
    if center_objects:
        description += "Center section: " + ", ".join(center_objects) + "\n"
    else:
        description += "Center section: Clear\n"
        
    if right_objects:
        description += "Right section: " + ", ".join(right_objects) + "\n"
    else:
        description += "Right section: Clear\n"
    
    # Speak the description
    speak_async(description)
    # Modify the last line to use rounded distance
    distance = read_distance()
    if distance is not None:
        speak_async(f"Distance is {round(distance)} centimeters")
    return description

def process_camera(input_source=0, desired_fps=30):
    cap = cv2.VideoCapture(input_source)

    if not cap.isOpened():
        print("Error: Could not open input source.")
        return

    # Set camera properties if the input source is a camera index
    if isinstance(input_source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, desired_fps)
    
    print("Processing input source. Press 'q' to quit.")

    frame_processor = FrameProcessor()
    frame_times = deque(maxlen=30)
    frame_count = 0
    start_time = time.time()
    last_key_press = 0  # Add this variable to track last key press time

    try:
        while True:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Add frame to processing queue, dropping old frames if necessary
            if frame_processor.frame_queue.full():
                try:
                    frame_processor.frame_queue.get_nowait()  # Remove old frame
                except:
                    pass
            frame_processor.frame_queue.put(frame)

            # Display processed frame if available
            if not frame_processor.result_queue.empty():
                display_frame = frame_processor.result_queue.get()
                cv2.imshow('Live Camera Feed', display_frame)

            key = cv2.waitKey(1) & 0xFF
            current_time = time.time()
            
            if key == ord('q'):
                break
            elif key == ord('d') and frame_processor.last_results:
                description = generate_scene_description(frame, frame_processor.last_results)
                print("\n" + description)
            elif key == ord('a') and (current_time - last_key_press) > 0.5:  # 0.5 second delay between keypresses
                last_key_press = current_time
                # Get current frame info directly from processor
                frame_info = frame_processor.frame_info
                if frame_info:
                    # Announce wall first if detected
                    if 'wall_message' in frame_info and frame_info['wall_message']:
                        speak_async(frame_info['wall_message'])
                    # Then proceed with other announcements
                    elif 'path_message' in frame_info:
                        speak_async(frame_info['path_message'])
                    if 'sticker_message' in frame_info and frame_info['sticker_message']:
                        speak_async(frame_info['sticker_message'])
                    elif 'distance' in frame_info and frame_info['distance'] is not None:
                        distance = frame_info['distance']  # Already rounded in process_frame
                        if distance < DISTANCE_THRESHOLD:
                            speak_async(f"Warning: obstacle {distance} centimeters ahead")

    finally:
        frame_processor.stop()
        cap.release()
        cv2.destroyAllWindows()
        if ser is not None:
            ser.close()
            print("Serial connection closed")
        # Stop the speech thread
        speech_queue.put(None)
        speech_thread.join()

    end_time = time.time()
    elapsed_time = end_time - start_time
    avg_fps = frame_count / elapsed_time
    # display if debug is enabled
    if debug:
        print(f"Average FPS (Input): {avg_fps:.2f}")

# GUI to start the blind program
# Allows user to select input camera, start/stop the program
# Display the output of the program
# Has an optional debug mode with additional information (FPS, terminal piping)

def start_blind():
    # Implement the GUI here
    root = tk.Tk()
    root.title("Blind Navigation System")
    root.geometry("800x600")

    # Variables for input source and serial port
    input_source_var = tk.StringVar(value="0")
    serial_port_var = tk.StringVar(value="COM6")

    # Create a frame for input source selection
    source_frame = tk.Frame(root)
    source_frame.pack(pady=5)

    tk.Label(source_frame, text="Input Source:").grid(row=0, column=0, padx=5, pady=5)
    input_source_entry = tk.Entry(source_frame, textvariable=input_source_var, width=20)
    input_source_entry.grid(row=0, column=1, padx=5, pady=5)

    def browse_file():
        file_path = filedialog.askopenfilename(title="Select Video File",
                                               filetypes=[("Video Files", "*.mp4;*.avi;*.mov"), ("All Files", "*.*")])
        if file_path:
            input_source_var.set(file_path)

    browse_button = tk.Button(source_frame, text="Browse", command=browse_file)
    browse_button.grid(row=0, column=2, padx=5, pady=5)

    # Create a frame for serial port selection
    port_frame = tk.Frame(root)
    port_frame.pack(pady=5)

    tk.Label(port_frame, text="Serial Port:").grid(row=0, column=0, padx=5, pady=5)
    serial_port_entry = tk.Entry(port_frame, textvariable=serial_port_var, width=20)
    serial_port_entry.grid(row=0, column=1, padx=5, pady=5)

    # Create a frame for the video feed
    video_frame = tk.Frame(root)
    video_frame.pack(pady=10)

    # Create a label for the video feed
    video_label = tk.Label(video_frame)
    video_label.pack()

    # Create a frame for the buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)

    # Function to run process_camera in a separate thread
    def start_program():
        input_source = input_source_var.get()
        serial_port = serial_port_var.get()

        # Update the global serial port
        global ser
        ser = try_open_serial(port=serial_port)
        if not ser:
            messagebox.showerror("Error", f"Could not connect to {serial_port}")
            return

        # Determine if input_source is a digit (camera index) or a file path
        try:
            input_source_int = int(input_source)
            source = input_source_int
        except ValueError:
            source = input_source

        # Start the camera processing in a new thread
        threading.Thread(target=process_camera, args=(source, desired_fps), daemon=True).start()

    # Create a button to start the program
    start_button = tk.Button(button_frame, text="Start Program", command=start_program)
    start_button.grid(row=0, column=0, padx=10)

    # Create a button to stop the program
    stop_button = tk.Button(button_frame, text="Stop Program", command=lambda: root.quit())
    stop_button.grid(row=0, column=1, padx=10)

    # Create a frame for the debug options
    debug_frame = tk.Frame(root)
    debug_frame.pack(pady=10)

    # Create a variable to store the debug mode
    debug_mode = tk.IntVar()
    debug_mode.set(0)

    # Create a check button for debug mode
    debug_check = tk.Checkbutton(debug_frame, text="Debug Mode", variable=debug_mode)
    debug_check.pack()

    # Function to update the video feed
    def update_video():
        # This part can be customized as needed
        # Currently left empty to focus on input source and serial port
        video_label.after(10, update_video)

    # Start the video feed
    update_video()

    # Start the GUI event loop
    root.mainloop()

# Simplify the main block
if __name__ == "__main__":
    try:
        start_blind()
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    finally:
        # Ensure the speech thread is properly terminated
        speech_queue.put(None)
        speech_thread.join()