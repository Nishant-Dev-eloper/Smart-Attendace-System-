import cv2
import os
import numpy as np
import pandas as pd

def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

# Create required directories
create_directory_if_not_exists("TrainingImage")
create_directory_if_not_exists("TrainingImageLabel")
create_directory_if_not_exists("StudentDetails")

# Initialize camera
print("Opening camera...")
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cam.isOpened():
    print("Error: Could not open camera!")
    exit()

# Set camera properties
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FPS, 30)

# Get user details
enrollment = input("Enter your Enrollment number: ")
name = input("Enter your Name: ")

# Create directory for user
user_dir = os.path.join("TrainingImage", f"{enrollment}_{name}")
create_directory_if_not_exists(user_dir)

# Initialize face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
if face_cascade.empty():
    print("Error: Could not load face cascade classifier!")
    exit()

print("\nStarting image capture...")
print("Look at the camera and move your face slightly to capture different angles")
print("Press 'q' to quit capturing\n")

image_count = 0
required_images = 50

while image_count < required_images:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame!")
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
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Save the face image
        face_img = gray[y:y+h, x:x+w]
        image_path = os.path.join(user_dir, f"{name}_{enrollment}_{image_count+1}.jpg")
        cv2.imwrite(image_path, face_img)
        image_count += 1
        print(f"Captured image {image_count}/{required_images}", end='\r')
    
    # Show remaining count
    cv2.putText(frame, f"Images: {image_count}/{required_images}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display frame
    cv2.imshow('Registration', frame)
    
    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save user details to CSV
csv_path = os.path.join("StudentDetails", "studentdetails.csv")
if not os.path.exists(csv_path):
    pd.DataFrame(columns=["Enrollment", "Name"]).to_csv(csv_path, index=False)

df = pd.read_csv(csv_path)
if enrollment not in df['Enrollment'].values:
    new_row = pd.DataFrame({"Enrollment": [enrollment], "Name": [name]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print("\nUser details saved to database")

print("\nImage capture completed!")
print(f"Captured {image_count} images")

# Cleanup
cam.release()
cv2.destroyAllWindows()

# Train the model
print("\nTraining the model...")
from trainImage import TrainImage
TrainImage("haarcascade_frontalface_default.xml", "TrainingImage", 
          os.path.join("TrainingImageLabel", "Trainner.yml"), None, print)

print("\nRegistration and training completed!")
print("Now you can run test_recognition.py to test face recognition") 