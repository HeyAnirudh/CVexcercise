import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#st.title("Welcome to CVExcercise Demo")
Activity=["Main","Posedetection","Excercise","References"]
choice=st.sidebar.selectbox("Select",Activity)
#@st.cache(suppress_st_warning=True)
def pose_mediapipe():
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture("data\PushUp.mp4")
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
        
           

    

def calculate_angle(a,b,c):
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

def Angle_calulator():
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture("data\PushUp.mp4")
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
            imagey = cv2.cvtColor(imagey, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                
                # Visualize angle
                cv2.putText(imagey, str(angle), 
                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                        
            except:
                pass
            
            
            # Render detections
            mp_drawing.draw_landmarks(imagey, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            FRAME_WINDOW.image(imagey)
    
def curl_calculator():
    FRAME_WINDOW = st.image([])
    Video="data\\1020355081-preview.mp4"
    Camera=0
    vid=Video
    cam=Camera
    yo=st.selectbox("pick source",(vid,cam))
    cap = cv2.VideoCapture(yo)
    start_demo = st.checkbox('Start Demo')
    counter = 0 
    stage = None

    ## Setup mediapipe instance
    if start_demo:
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
                    #imagey= cv2.cvtColor(imagey, cv2.COLOR_RGB2BGR)
                    
                    # Extract landmarks
                    try:
                        landmarks = results.pose_landmarks.landmark
                        
                        # Get coordinates
                        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        
                        # Calculate angle
                        angle = calculate_angle(shoulder, elbow, wrist)
                        
                        # Visualize angle
                        cv2.putText(imagey, str(angle), 
                                    tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                            )
                        
                        # Curl counter logic
                        if angle > 160:
                            stage = "down"
                        if angle < 30 and stage =='down':
                            stage="up"
                            counter +=1
                            print(counter)
                                
                    except:
                        pass
                
                    # Render curl counter
                    # Setup status box
                    cv2.rectangle(imagey, (0,0), (225,73), (245,117,16), -1)
                    
                    # Rep data
                    cv2.putText(imagey, 'REPS', (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(imagey, str(counter), 
                                (10,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                    
                    # Stage data
                    cv2.putText(imagey, 'STAGE', (65,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(imagey, stage, 
                                (60,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                    
                    
                    # Render detections
                    mp_drawing.draw_landmarks(imagey, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                            )        
                        
                    
                    FRAME_WINDOW.image(imagey)
    else:
        st.write("band hogaya")
            


            


if choice=="Main":
    st.title("Welcome to CVExcercise Demo")
    st.text("Built with opencv and streamlit")
    st.success("By Anirudh soni and Keshav rao")
if choice=="Posedetection":
    st.title("Pose detection with Mediapipe Library")
    st.subheader("Person/pose Detection Model (BlazePose Detector)")

    st.write("The detector is inspired by our own lightweight BlazeFace model, used in MediaPipe Face Detection, as a proxy for a person detector. It explicitly predicts two additional virtual keypoints that firmly describe the human body center, rotation and scale as a circle. Inspired by Leonardo’s Vitruvian man, we predict the midpoint of a person’s hips, the radius of a circle circumscribing the whole person, and the incline angle of the line connecting the shoulder and hip midpoints.")
    #certi = cv2.imread("C:\\Users\\Admin\\Desktop\\github\\CVexcercise\\poseestimation\\data\\pose_tracking_detector_vitruvian_man.png")
    #m=cv2.imread("data\\pose_tracking_detector_vitruvian_man.png")
    #st.image(m)
    #n=cv2.imread("data/3j8BPdc.png")
    #st.image(n)
    st.subheader("Try Demo")
    pose_mediapipe()
    st.subheader("Understanding the code!")
    st.markdown("--> Installing required library")
    code="pip install streamlit \npip install mediapipe \npip install opencv-python \npip install numpy"
    st.code(code)
    
if choice=="Excercise":
    st.title("Excercise Demo")
    st.subheader("Understanding Cordinates")
    st.write("The k-NN algorithm used for pose classification requires a feature vector representation of each sample and a metric to compute the distance between two such vectors to find the nearest pose samples to a target one.To convert pose landmarks to a feature vector, we use pairwise distances between predefined lists of pose joints, such as distances between wrist and shoulder, ankle and hip, and two wrists. Since the algorithm relies on distances, all poses are normalized to have the same torso size and vertical torso orientation before the conversion.")
    st.video("data/pose_world_landmarks.mp4")
    #z=cv2.imread("data\\pose_classification_pairwise_distances.png")
    #st.image(z)
    curl_calculator()

if choice=="References":
    st.title("References")
    st.write("An Overview of Human Pose Estimation with Deep Learning [Link](https://www.kdnuggets.com/2019/06/human-pose-estimation-deep-learning.html)")
    st.write("Posture Detection using PoseNet with Real-time Deep Learning project [Link](https://www.analyticsvidhya.com/blog/2021/09/posture-detection-using-posenet-with-real-time-deep-learning-project/)")
    st.write("Human Pose Estimation : Simplified [Link](https://towardsdatascience.com/human-pose-estimation-simplified-6cfd88542ab3)")
    