import numpy as np
import warnings
from io import BytesIO

# Suppress FutureWarnings for numpy deprecations
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    if not hasattr(np, 'object'):
        np.object = object
    if not hasattr(np, 'bool'):
        np.bool = bool
    if not hasattr(np, 'typeDict'):
        np.typeDict = np.sctypeDict

import rospy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw
from gtts import gTTS
import pygame
import time
import os
from sensor_msgs.msg import Image as ROSImage
import cv2
import NotCvBridge  # Import the custom bridge

# Define the neural network
class Net(nn.Module):
    def init(self, num_classes):
        super(Net, self).init()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_model(path, num_classes):
    model = Net(num_classes)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_image(model, image, classes):
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return classes[predicted.item()]

def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    filename = 'output.mp3'
    if os.path.exists(filename):
        os.remove(filename)
    tts.save(filename)
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(1)
    pygame.mixer.music.unload()
    pygame.mixer.quit()

def image_to_voice(image, model, classes):
    print("image to voice called")
    predicted_class = predict_image(model, image, classes)
    if predicted_class is None:
        text_to_speech("no hand detected")
    else:
        print(f'The predicted class for the image is: {predicted_class}')
        text_to_speech(predicted_class)


# MediaPipe hand module
import mediapipe as mp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def detect_and_crop_palm(frame):
    frame_rgb = frame.convert('RGB')
    results = hands.process(np.array(frame_rgb))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            width, height = frame.size
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * width) + 30
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * width) - 30
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * height) + 30
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * height) - 30
            x_max = min(width, max(0, x_max))
            x_min = max(0, min(width, x_min))
            y_max = min(height, max(0, y_max))
            y_min = max(0, min(height, y_min))
            cropped_palm = frame.crop((x_min, y_min, x_max, y_max))
            return cropped_palm
    return None

def draw_finger_lines_and_joints(frame, hand_landmarks):
    draw = ImageDraw.Draw(frame)
    line_color = (0x80, 0x78, 0x00)
    joint_color = (0x00, 0x00, 0x78)
    width, height = frame.size
    for hand_landmark in hand_landmarks:
        for connection in mp_hands.HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            start_point = hand_landmark.landmark[start_idx]
            end_point = hand_landmark.landmark[end_idx]
            start_coords = (int(start_point.x * width), int(start_point.y * height))
            end_coords = (int(end_point.x * width), int(end_point.y * height))
            draw.line([start_coords, end_coords], fill=line_color, width=2)
        for landmark in hand_landmark.landmark:
            coords = (int(landmark.x * width), int(landmark.y * height))
            draw.ellipse([coords[0]-5, coords[1]-5, coords[0]+5, coords[1]+5], fill=joint_color)

class HandDetectionNode:
    def init(self):
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", ROSImage, self.image_callback)
        self.model = load_model('./dataset_final.pth', 6)
        self.classes = ['not detected', 'call me', 'hello', 'i love you', 'not ok', 'ok']
        self.start_time = time.time()

    def image_callback(self, data):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = NotCvBridge.imgmsg_to_cv2(data)
            
            # Convert OpenCV image to PIL image
            frame = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            # Detect and crop palm
            cropped_palm = detect_and_crop_palm(frame)
            #if cropped_palm is not None:
                #cropped_palm.show('Cropped Palm')<<--- this line is commented out because linux cant print out alot of images without crashing
            
            # Process the image to detect hand landmarks
            results = hands.process(np.array(frame))
            if results.multi_hand_landmarks:
                draw_finger_lines_and_joints(frame, results.multi_hand_landmarks)
            #frame.show('Hand Detection') <<--- this line is commented out because linux cant print out alot of images without crashing
            
            # If more than 5 seconds have passed, predict and convert text to speech
            if time.time() - self.start_time > 2:
                if cropped_palm is not None:
                    image_to_voice(cropped_palm, self.model, self.classes)
                self.start_time = time.time()
        
        except Exception as e:
            rospy.logerr("Error in image_callback: %s", str(e))

if name == 'main':
    rospy.init_node('hand_detection_node', anonymous=True)
    hand_detection_node = HandDetectionNode()
    rospy.spin()