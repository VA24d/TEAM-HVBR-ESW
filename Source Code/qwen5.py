#, with distance averaged and walls detected 
# gui is working without issues
#added door detection green color at the center , left right
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
        print("YOLOv11 model loaded successfully.")
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

def process_frame(frame, processor):
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
                        # Initialize color variable
                        color = None
                        
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

                        # Add green color mask for door detection
                        lower_green = np.array([40, 40, 40])  # HSV values for green
                        upper_green = np.array([80, 255, 255])
                        mask_green = cv2.inRange(hsv_roi, lower_green, upper_green)
                        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
                        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

                        green_pixels = cv2.countNonZero(mask_green)

                        # Include green in color detection
                        if red_pixels > blue_pixels and red_pixels > green_pixels and red_pixels > total_pixels * 0.3:
                            color = 'red'
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        elif blue_pixels > red_pixels and blue_pixels > green_pixels and blue_pixels > total_pixels * 0.3:
                            color = 'blue'
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        elif green_pixels > total_pixels * 0.3:  # Threshold for green detection
                            color = 'green'
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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

        # Door detection logic - Update the messages
        door_detected = False
        door_message = ''
        door_display_message = ''
        door_location = ''  # New variable to track door location
        
        # Check for door (green color) in all regions
        for obj in colored_objects:
            x1, y1, x2, y2, color = obj
            obj_center_x = (x1 + x2) // 2
            
            if color == 'green' and distance is not None:
                door_detected = True
                # Determine door location based on object position
                if obj_center_x < left_boundary:
                    door_location = 'left'
                elif obj_center_x > right_boundary:
                    door_location = 'right'
                else:
                    door_location = 'center'
                
                # Create appropriate messages based on location
                if door_location == 'center':
                    door_display_message = f'Door detected ahead'
                    door_message = f'Door detected {round(distance)} centimeters ahead. Reach out to open the door'
                else:
                    door_display_message = f'Door detected on the {door_location}'
                    door_message = f'Door detected on the {door_location}'
                break

        # Update display messages
        if door_detected:
            cv2.putText(frame, door_display_message, (10, frame_height - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Update the processor's frame_info with door location
        processor.frame_info = {
            'path_message': path_message,
            'sticker_message': sticker_message if stickers_detected else None,
            'distance': rounded_distance if rounded_distance is not None else None,
            'wall_message': wall_message if wall_detected else None,
            'door_message': door_message if door_detected else None,
            'door_display_message': door_display_message if door_detected else None,
            'door_location': door_location if door_detected else None  # Add door location to frame info
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
            # Check for shutdown event at start of loop
            if hasattr(frame_processor, 'shutdown_event') and frame_processor.shutdown_event.is_set():
                break
                
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
            
            if key == ord('q') or (hasattr(frame_processor, 'shutdown_event') and frame_processor.shutdown_event.is_set()):
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
                    if 'door_message' in frame_info and frame_info['door_message']:
                        speak_async(frame_info['door_message'])
                    elif 'wall_message' in frame_info and frame_info['wall_message']:
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
        # Cleanup in reverse order of creation
        frame_processor.stop()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if ser is not None:
            ser.close()

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

class BlindNavigationGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Blind Navigation System")
        self.root.geometry("1024x768")
        self.root.configure(bg='#f0f0f0')
        
        # Create scrollable canvas
        self.canvas = tk.Canvas(self.root, bg='#f0f0f0')
        self.scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg='#f0f0f0')
        
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Variables
        self.input_source_var = tk.StringVar(value="0")
        self.serial_port_var = tk.StringVar(value="COM6")
        self.is_running = False
        self.cap = None
        self.preview_running = False
        self.closing = False  # Add this flag
        self.process_thread = None
        self.shutdown_event = threading.Event()
        self.quit_flag = False  # Add this flag
        
        self.create_gui()
        
        # Configure scrolling
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Pack scrollbar and canvas
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Bind mouse wheel
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.root.bind('<w>', self.start_program_on_w)
        speak_async("Blind Navigation System initalized. Press 'w' to start the program.")

    def start_program_on_w(self, event):
        """Start the program when 'w' key is pressed"""
        self.start_program()
        
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
    def create_gui(self):
        # Main container with padding
        main_container = tk.Frame(self.scrollable_frame, bg='#f0f0f0')
        main_container.pack(padx=20, pady=20, fill='both', expand=True)
        
        # Input Source Section with Radio Buttons
        source_frame = self.create_labeled_frame(main_container, "Camera Selection")
        source_frame.pack(fill='x', pady=(0, 10))
        
        tk.Radiobutton(source_frame, 
                      text="Default Camera (0)", 
                      variable=self.input_source_var, 
                      value="0",
                      bg='#f0f0f0').pack(side='left', padx=5)
        
        tk.Radiobutton(source_frame, 
                      text="External Camera (1)", 
                      variable=self.input_source_var, 
                      value="1",
                      bg='#f0f0f0').pack(side='left', padx=5)
        
        # Serial Port Section
        port_frame = self.create_labeled_frame(main_container, "Serial Port Configuration")
        port_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(port_frame, text="Port:", bg='#f0f0f0').pack(side='left', padx=5)
        tk.Entry(port_frame, textvariable=self.serial_port_var, width=30).pack(side='left', padx=5)
        
        # Video Preview Section
        preview_frame = self.create_labeled_frame(main_container, "Video Preview")
        preview_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        self.video_label = tk.Label(preview_frame, bg='black')
        self.video_label.pack(padx=10, pady=10, fill='both', expand=True)
        
        # Controls Section - Now in a fixed position
        control_frame = self.create_labeled_frame(main_container, "Controls")
        control_frame.pack(fill='x', pady=(0, 10))
        
        button_frame = tk.Frame(control_frame, bg='#f0f0f0')
        button_frame.pack(fill='x', padx=10, pady=5)
        
        self.preview_btn = tk.Button(button_frame, 
                                   text="Start Preview",
                                   command=self.toggle_preview,
                                   bg='#4a90e2', 
                                   fg='white', 
                                   relief='flat', 
                                   padx=20)
        self.preview_btn.pack(side='left', padx=5)
        
        self.start_btn = tk.Button(button_frame, 
                                 text="Start Program",
                                 command=self.start_program,
                                 bg='#2ecc71', 
                                 fg='white', 
                                 relief='flat', 
                                 padx=20)
        self.start_btn.pack(side='left', padx=5)
        
        self.stop_btn = tk.Button(button_frame, 
                                text="Stop",
                                command=self.stop_program,
                                bg='#e74c3c', 
                                fg='white', 
                                relief='flat', 
                                padx=20,
                                state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
        # Status Section
        status_frame = self.create_labeled_frame(main_container, "Status")
        status_frame.pack(fill='x')
        
        self.status_label = tk.Label(status_frame, 
                                   text="Ready", 
                                   bg='#f0f0f0',
                                   wraplength=900)  # Allow status text to wrap
        self.status_label.pack(padx=5, pady=5)

    # Remove the browse_file method since we're using radio buttons now

    def update_preview(self):
        if self.preview_running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # Resize frame to fit preview window while maintaining aspect ratio
                preview_width = 800
                preview_height = 600
                h, w = frame.shape[:2]
                scaling = min(preview_width/w, preview_height/h)
                new_w, new_h = int(w*scaling), int(h*scaling)
                
                frame = cv2.resize(frame, (new_w, new_h))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = ImageTk.PhotoImage(image=img)
                
                self.video_label.config(image=img)
                self.video_label.image = img
                
                if self.preview_running:  # Check if preview is still running
                    self.root.after(30, self.update_preview)
            else:
                self.stop_preview()

    # ...rest of the existing methods...

    def start_preview(self):
        try:
            source = int(self.input_source_var.get())  # Always convert to int
            
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise Exception(f"Could not open camera {source}")
            
            self.preview_running = True
            self.preview_btn.config(text="Stop Preview", bg='#e74c3c')
            self.status_label.config(text="Preview running...")
            self.update_preview()
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not start preview: {str(e)}")
            self.stop_preview()

    def toggle_preview(self):
        if not self.preview_running:
            self.start_preview()
        else:
            self.stop_preview()

    def stop_preview(self):
        self.preview_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.preview_btn.config(text="Start Preview", bg='#4a90e2')
        self.video_label.config(image='')
        self.status_label.config(text="Preview stopped")

    def start_program(self):
        if not self.is_running:
            try:
                # Stop preview if running
                self.stop_preview()
                
                input_source = self.input_source_var.get()
                serial_port = self.serial_port_var.get()
                
                # Update serial connection
                global ser
                ser = try_open_serial(port=serial_port)
                if not ser:
                    raise Exception(f"Could not connect to {serial_port}")
                
                source = int(input_source) if input_source.isdigit() else input_source
                
                # Start processing in new thread
                self.process_thread = threading.Thread(
                    target=process_camera,
                    args=(source, desired_fps),
                    daemon=True
                )
                self.process_thread.start()
                
                print("Program started")
                self.is_running = True
                self.start_btn.config(state='disabled')
                self.stop_btn.config(state='normal')
                self.preview_btn.config(state='disabled')
                self.status_label.config(text="Program running...")
                
            except Exception as e:
                messagebox.showerror("Error", str(e))
                self.stop_program()

    def stop_program(self):
        if self.is_running:
            self.quit_flag = True  # Set quit flag
            # Simulate 'q' key press in OpenCV window
            cv2.waitKey(1) & 0xFF
            
            # Stop preview if running
            self.stop_preview()
            
            # Wait briefly for cleanup
            time.sleep(0.5)
            
            # Close all OpenCV windows
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            
            # Cleanup serial connection
            global ser
            if ser is not None:
                ser.close()
                ser = None
            
            self.is_running = False
            self.start_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            self.preview_btn.config(state='normal')
            self.status_label.config(text="Program stopped")

    def on_closing(self):
        try:
            # Stop the program if running
            if self.is_running:
                self.stop_program()
            
            # Final cleanup
            cv2.destroyAllWindows()
            if ser is not None:
                ser.close()
            
            # Clear speech queue
            while not speech_queue.empty():
                speech_queue.get_nowait()
            speech_queue.put(None)
            
            # Destroy root window
            self.root.quit()
            self.root.destroy()
            
        except Exception as e:
            print(f"Error during closing: {e}")
        finally:
            # Force exit
            import os
            os._exit(0)

    def force_cleanup(self):
        """Force cleanup of all resources"""
        try:
            # Stop any running preview
            self.preview_running = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            # Force close all OpenCV windows
            cv2.destroyAllWindows()
            for i in range(1):  # Workaround for Windows
                cv2.waitKey(1)

            # Clean up serial connection
            global ser
            if ser is not None:
                ser.close()
                ser = None

            # Clean up speech queue
            while not speech_queue.empty():
                speech_queue.get_nowait()
            speech_queue.put(None)

        except Exception as e:
            print(f"Error in force_cleanup: {e}")

    def on_closing(self):
        if self.closing:  # Prevent multiple closing attempts
            return
            
        self.closing = True
        try:
            # Update status
            self.status_label.config(text="Shutting down...")
            self.root.update()

            # Stop preview and main program
            self.stop_preview()
            self.stop_program()
            
            # Force cleanup
            self.force_cleanup()
            
            # Schedule final cleanup and destroy
            self.root.after(100, self._final_cleanup)
            
        except Exception as e:
            print(f"Error during closing: {e}")
            self._force_exit()

    def _final_cleanup(self):
        """Final cleanup and window destruction"""
        try:
            # Final attempt to cleanup
            self.force_cleanup()
            
            # Destroy root window
            self.root.quit()
            self.root.destroy()
            
        except Exception as e:
            print(f"Error in final cleanup: {e}")
        finally:
            self._force_exit()

    def _force_exit(self):
        """Force exit the program"""
        try:
            import os
            os._exit(0)  # Force exit
        except:
            import sys
            sys.exit(0)

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"Error in mainloop: {e}")
            self._force_exit()

    def create_labeled_frame(self, parent, text):
        """Create a labeled frame with consistent styling"""
        frame = tk.LabelFrame(
            parent,
            text=text,
            bg='#f0f0f0',
            font=('Arial', 10, 'bold'),
            padx=10,
            pady=5,
            relief=tk.GROOVE,
            borderwidth=1
        )
        return frame

# Update the main block to use the new GUI class
if __name__ == "__main__":
    try:
        app = BlindNavigationGUI()
        app.run()
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        # Ensure the speech thread is properly terminated
        speech_queue.put(None)
        speech_thread.join()