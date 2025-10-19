import cv2
import os
import numpy as np
import pandas as pd

def test_face_recognition():
    # Paths
    haar_cascade_path = "haarcascade_frontalface_default.xml"
    model_path = os.path.join("TrainingImageLabel", "Trainner.yml")
    student_data_path = os.path.join("StudentDetails", "studentdetails.csv")

    # Check if model exists
    if not os.path.exists(model_path):
        print("Error: Trained model not found. Please train the model first.")
        return

    # Load the trained model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.read(model_path)
        print("Successfully loaded the trained model")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    # Load student data
    try:
        df = pd.read_csv(student_data_path)
        print("Loaded student data:")
        print(df)
    except Exception as e:
        print(f"Error loading student data: {str(e)}")
        return

    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)
    if face_cascade.empty():
        print("Error: Could not load face cascade classifier")
        return

    # Initialize camera
    print("Opening camera...")
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        print("Error: Could not open camera")
        return

    # Set camera properties
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FPS, 30)

    print("Starting face recognition... Press 'q' to quit")
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            # Recognize face
            face_roi = gray[y:y+h, x:x+w]
            id_predicted, confidence = recognizer.predict(face_roi)
            
            # Get name from student data
            try:
                name = df.loc[df['Enrollment'] == id_predicted]['Name'].values[0]
            except:
                name = "Unknown"

            # Draw rectangle and put text
            if confidence < 80:  # Confidence threshold
                color = (0, 255, 0)  # Green for recognized
                text = f"{name} ({confidence:.2f}%)"
            else:
                color = (0, 0, 255)  # Red for unknown
                text = f"Unknown ({confidence:.2f}%)"

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Show frame
        cv2.imshow('Face Recognition', frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_face_recognition() 