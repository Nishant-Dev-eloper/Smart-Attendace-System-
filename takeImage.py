import csv
import os
import cv2
import numpy as np
from PIL import Image
from face_recognition_dl import FaceRecognitionDL

def TakeImage(enrollment, name, trainimage_path, message, err_screen, text_to_speech):
    if not enrollment or not name:
        err_screen()
        return
        
    try:
        # Initialize face recognition system
        model_path = os.path.join(os.getcwd(), "models")
        os.makedirs(model_path, exist_ok=True)
        print("\nInitializing face recognition system...")
        face_recognizer = FaceRecognitionDL(model_path=model_path)
        print("Face recognition system initialized")
        
        # Initialize camera
        print("\nInitializing camera...")
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cam.isOpened():
            error_msg = "Error: Could not open camera. Please check if camera is connected."
            print(error_msg)
            if message:
                message.configure(text=error_msg)
            if text_to_speech:
                text_to_speech(error_msg)
            return
        print("Camera initialized successfully")
            
        # Set camera properties
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cam.set(cv2.CAP_PROP_FPS, 30)
        
        # Create directory for user
        user_dir = os.path.join(trainimage_path, f"{enrollment}_{name}")
        os.makedirs(user_dir, exist_ok=True)
        print(f"\nSaving images to: {user_dir}")
        
        image_count = 0
        required_images = 20  # Number of images to capture
        
        print("\nStarting image capture...")
        print("Look at the camera and move your face slightly to capture different angles")
        print("Press 'q' to quit\n")
        
        if message:
            message.configure(text="Starting image capture... Please look at the camera")
        
        while image_count < required_images:
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame")
                continue
                
            # Make a copy for display
            display_frame = frame.copy()
            
            # Detect faces using MTCNN
            face_locations = face_recognizer.get_face_locations(frame)
            
            if face_locations:
                for (top, right, bottom, left) in face_locations:
                    # Extract and save face image
                    face_img = frame[top:bottom, left:right]
                    image_path = os.path.join(
                        user_dir,
                        f"{enrollment}_{name}_{image_count+1}.jpg"
                    )
                    cv2.imwrite(image_path, face_img)
                    image_count += 1
                    
                    # Draw rectangle around face
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # Show progress
                    text = f"Captured: {image_count}/{required_images}"
                    y = top - 10 if top - 10 > 10 else top + 10
                    cv2.putText(display_frame, text, (left, y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                    
                    print(f"Captured image {image_count}/{required_images}", end='\r')
                    
                    if message:
                        message.configure(text=f"Captured {image_count} of {required_images} images")
                    
                    if image_count >= required_images:
                        break
            else:
                # Show message if no face detected
                cv2.putText(display_frame, "No face detected - Please look at the camera", (30, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if message:
                    message.configure(text="No face detected - Please look at the camera")
            
            # Show frame
            cv2.imshow("Capturing Training Images", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cam.release()
        cv2.destroyAllWindows()
        
        if image_count > 0:
            # Save student details
            csv_path = os.path.join("StudentDetails", "studentdetails.csv")
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            
            # Check if file exists and create with header if it doesn't
            if not os.path.exists(csv_path):
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Enrollment', 'Name'])
            
            # Append student details
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([enrollment, name])
            
            success_msg = f"Successfully captured {image_count} images for {name} (Enrollment: {enrollment})"
            print(f"\n{success_msg}")
            if message:
                message.configure(text=success_msg)
            if text_to_speech:
                text_to_speech(success_msg)
                
            return True
        else:
            error_msg = "No images were captured. Please try again."
            print(f"\n{error_msg}")
            if message:
                message.configure(text=error_msg)
            if text_to_speech:
                text_to_speech(error_msg)
            return False
                
    except Exception as e:
        error_msg = f"Error capturing images: {str(e)}"
        print(error_msg)
        if message:
            message.configure(text=error_msg)
        if text_to_speech:
            text_to_speech(error_msg)
        return False
