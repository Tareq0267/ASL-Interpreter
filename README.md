Sure! Here's a `README.md` file for your project:

```markdown
# Hand Detection and Text-to-Speech Project

This project is developed for the subject Autonomous Robot by Dr. Zati. The aim is to create a system that detects hand gestures using a webcam and provides audio feedback using text-to-speech.

## Overview

The project consists of two main components:
1. Hand detection and gesture recognition using a neural network.
2. Text-to-speech conversion using GTTS (Google Text-to-Speech) for providing audio feedback.

The hand detection and gesture recognition is implemented in Python using PyTorch, OpenCV, and MediaPipe. The ROS (Robot Operating System) is used to integrate the different components and handle communication between nodes.

## Project Structure

The project is organized into the following directories and files:

```
.
├── hand_detection
│   ├── launch
│   │   └── hand_detection.launch
│   ├── scripts
│   │   └── hand_detection_node.py
│   └── src
│       └── NotCvBridge.py
├── usb_cam
│   ├── launch
│   │   └── usb_cam-test.launch
├── README.md
└── dataset_final.pth
```

### hand_detection

This directory contains the ROS package for hand detection and gesture recognition.

- **scripts/hand_detection_node.py**: The main ROS node for hand detection and gesture recognition.
- **src/NotCvBridge.py**: A custom script to replace `cv_bridge` for converting between ROS image messages and OpenCV images.
- **launch/hand_detection.launch**: A launch file to start the hand detection node in a Conda environment.

### usb_cam

This directory contains the ROS package for the USB camera.

- **launch/usb_cam-test.launch**: A launch file to start the USB camera node.

## Setup

### Prerequisites

- ROS (Robot Operating System) Noetic
- Conda (Anaconda or Miniconda)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/hand_detection_project.git
    cd hand_detection_project
    ```

2. Create and activate a Conda environment:

    ```bash
    conda create -n env38 python=3.8
    conda activate env38
    ```

3. Install necessary packages in the Conda environment:

    ```bash
    conda install pytorch torchvision cpuonly -c pytorch
    pip install gtts pygame opencv-python pillow mediapipe
    ```

4. Set up the Catkin workspace:

    ```bash
    mkdir -p ~/catkin_ws/src
    cd ~/catkin_ws/src
    ln -s /path/to/hand_detection_project/hand_detection .
    ln -s /path/to/hand_detection_project/usb_cam .
    cd ~/catkin_ws
    catkin_make
    source devel/setup.bash
    ```

## Running the Project

To run the project, use the following command:

```bash
roslaunch hand_detection hand_detection.launch
```

This will start both the USB camera node and the hand detection node.

## Issues Faced

### Compatibility with `cv_bridge`

During the integration process, compatibility issues with `cv_bridge` were encountered. `cv_bridge` is typically used to convert between ROS image messages and OpenCV images. To overcome these issues, a custom script named `NotCvBridge.py` was created to handle the conversions.

### Custom `NotCvBridge.py`

`NotCvBridge.py` performs the necessary conversions between ROS image messages and OpenCV images, allowing the project to proceed without relying on `cv_bridge`.

## Contributors

1. Mehran Gharooni Khoshkehbar (S2014607)
Pendar Tabatabaeemoshiri (S2029817)
Muhammad Tareq Adam Bin Ellias (U2001228)
Muhammad Adam Malique bin Zainal (U2102866)
Liu Yuxiang (S2193853)
Chu Wei Ming (U2102762)



## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
```

Feel free to modify the sections as per your requirements and add any additional information you think is necessary.