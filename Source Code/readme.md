# Blind Navigation System

A computer vision-based navigation system for visually impaired individuals using YOLO object detection, color detection, and distance sensing.

## Features

- Real-time object detection
- Distance measurement to obstacles
- Wall and door detection
- Path navigation assistance
- Directional guidance using color markers
- User-friendly GUI interface

## Prerequisites

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLO
- PyTTSx3
- Serial communication device (ESP32 or similar)
- Camera

## Installation

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
   in terminal

2. Download the YOLO model file (`yolo11x-seg.pt`) and place it in the project directory.
    ``` 
    https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt 
    ```
    ```
    https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-seg.pt
    ```

## Hardware Setup

1. Connect ESP32 (or similar device) with ultrasonic sensor on the proper pins.
2. Connect camera (built-in or external USB camera) using the appropriate port
3. Push attached code (```arduino.cpp```) to the arduino. (Check the GPIO pins)
```
trig_pin = 32
echo_pin = 33
```
4. Note the COM port for your serial device (default: COM6)

## Usage

### Starting the Program

Start the program by running ``` qwen5.py```

Use the default python interpreter in the terminal in the same folder along with running pip install the requirements and placing the YOLOv11 file. 


The GUI provides the following options:
- Camera Selection (0: Default, 1: External)
- Serial Port Configuration
- Video Preview
- Program Controls

Click "Start Program" or press 'w' to begin navigation.

### Control Keys

| Key | Function |
|-----|----------|
| 'w' | Start program |
| 'a' | Announce current situation |
| 'd' | Detailed scene description |
| 'q' | Quit program |

### Navigation Markers

| Marker Colors | Meaning |
|--------------|---------|
| Red/Red | Dead end or Exit South |
| Blue/Blue | Multiple paths or Exit North |
| Red/Blue | Turn right or Exit West |
| Blue/Red | Turn left or Exit East |
| Green | Door ahead |

## Troubleshooting

### Camera Issues
- Verify camera index (0 or 1)
- Ensure camera isn't in use by other programs

### Serial Communication
- Confirm correct COM port
- Check device connections
- Try resetting ESP32

### Speech Output
- Verify audio output device
- Check Windows speech settings
- Reinstall pyttsx3 if needed

## Notes

- Ensure good lighting conditions
- Keep camera lens clean
- Internet connection required for first run (YOLO model download)
