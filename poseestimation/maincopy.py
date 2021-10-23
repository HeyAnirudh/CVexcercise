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
    start_demo = st.checkbox('Start Demo')
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while start_demo:
            ret, frame = cap.read()
            
            # Recolor image to RGB
            imagey = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imagey.flags.writeable = False
        
            # Make detection
            results = pose.process(imagey)
        
            # Recolor back to BGR
            imagey.flags.writeable = True
            #imagey = cv2.cvtColor(imagey, cv2.COLOR_RGB2BGR)
            
            # Render detections
            mp_drawing.draw_landmarks(imagey, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            #cv2.imshow('Mediapipe Feed', image)
            FRAME_WINDOW.image(imagey)
        
           

    cap.release()
    cv2.destroyAllWindows()

def Excercise_code():

if choice=="Main":
    st.title("Welcome to CVExcercise Demo")
if choice=="Posedetection":
    st.title("Pose detection with Mediapipe Library")
    st.subheader("Person/pose Detection Model (BlazePose Detector)")

    st.write("The detector is inspired by our own lightweight BlazeFace model, used in MediaPipe Face Detection, as a proxy for a person detector. It explicitly predicts two additional virtual keypoints that firmly describe the human body center, rotation and scale as a circle. Inspired by Leonardo’s Vitruvian man, we predict the midpoint of a person’s hips, the radius of a circle circumscribing the whole person, and the incline angle of the line connecting the shoulder and hip midpoints.")
    certi = cv2.imread("C:\\Users\\Admin\\Desktop\\poseestimation\\pose_tracking_detector_vitruvian_man.png")
    m=cv2.imread("C:\\Users\\Admin\\Desktop\\poseestimation\\pose_tracking_detector_vitruvian_man.png")
    st.image(m)
    n=cv2.imread("C:/Users/Admin/Desktop/poseestimation/3j8BPdc.png")
    st.image(n)
    st.subheader("Try Demo")
    pose_mediapipe()

if choice=="Excercise":
    st.title("Excercise Demo")
    st.subheader("Understanding Cordinates")
    st.write("The k-NN algorithm used for pose classification requires a feature vector representation of each sample and a metric to compute the distance between two such vectors to find the nearest pose samples to a target one.To convert pose landmarks to a feature vector, we use pairwise distances between predefined lists of pose joints, such as distances between wrist and shoulder, ankle and hip, and two wrists. Since the algorithm relies on distances, all poses are normalized to have the same torso size and vertical torso orientation before the conversion.")
    st.video("C:\\Users\\Admin\\Desktop\\poseestimation\\data\\pose_world_landmarks.mp4")
    z=cv2.imread("C:\\Users\\Admin\\Desktop\\poseestimation\\data\\pose_classification_pairwise_distances.png")
    st.image(z)
