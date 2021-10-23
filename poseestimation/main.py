import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#st.title("Welcome to CVExcercise Demo")
Activity=["Main","Posedetection","Excercise","References"]
choice=st.sidebar.selectbox("Select",Activity)

def pose_mediapipe():
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            imagey = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imagey.flags.writeable = False
        
            # Make detection
            results = pose.process(imagey)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(imagey, cv2.COLOR_RGB2BGR)
            
            # Render detections
            mp_drawing.draw_landmarks(imagey, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            #cv2.imshow('Mediapipe Feed', image)
            FRAME_WINDOW.image(imagey)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if choice=="Main":
    st.title("Welcome to CVExcercise Demo")
if choice=="Posedetection":
    st.title("Pose detection with Mediapipe Library")
    
    pose_mediapipe()

