"""
Real-time Object Detection with Audio Feedback
Copyright (C) [Year] [Your Name]

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import cv2
from ultralytics import YOLO
import pyttsx3
import torch
import time


engine = pyttsx3.init()
confidence_threshold = 0.8
device = 'cuda' if torch.cuda.is_available() else 'cpu'

last_speech_time = 0
speech_interval = 5

def play_audio(label):
    engine.say(label)
    engine.runAndWait()

def move_left(text="move left"):
    engine.say(text)
    engine.runAndWait()

def move_right(text="move right"):
    engine.say(text)
    engine.runAndWait()


model = YOLO('yolov8n.pt').to(device)  


cap = cv2.VideoCapture(0) 


object_labels = set()

while True:
 
    ret, frame = cap.read()
    if not ret:
        break

   
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

 
    results = model(frame_rgb)

    obstacle_in_center=False
    obstacle_in_left=False
    obstacle_in_right=False

    
    detected_objects = []
    
    
    for result in results:
  
        annotated_frame = result.plot() 

        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        
      
        if result.boxes:
            for box in result.boxes:
                b = box.xyxy[0]  
                c = box.cls
                x,y,h,w = b
                if x < 640//3:
                    obstacle_in_left=True
                elif x+w > 2 * 640//3:
                    obstacle_in_right=True
                else:
                    obstacle_in_center=True
                class_id = int(box.cls.item())  
                confidence = box.conf.item()    
                if confidence >= confidence_threshold:
                    label = result.names[class_id]
                    if label:

                        detected_objects.append(label)
                        current_time = time.time()
                        if detected_objects and (current_time - last_speech_time > speech_interval):
                            object_text = "Detected: " + ", ".join(detected_objects)
                            play_audio(object_text) 
                            last_speech_time = current_time
                            if obstacle_in_center:
                                if not obstacle_in_left:
                                    move_left()
                                elif not obstacle_in_right:
                                    move_right()
                                else:
                                    play_audio("Obstacle ahead, can't move")
                            elif obstacle_in_left:
                                move_right()
                            elif obstacle_in_right:
                                move_left()

    cv2.imshow('YOLOv8 Object Detection', annotated_frame_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
