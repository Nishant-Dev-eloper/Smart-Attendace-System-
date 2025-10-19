import cv2
import numpy as np
from face_recognition_dl import FaceRecognitionDL
import os

def test_camera_and_detection():
    print("\nTesting camera and face detection...")
    
    # Initialize face recognition system
    model_path = os.path.join(os.getcwd(), "models")
    os.makedirs(model_path, exist_ok=True)
    print(f"Using model path: {model_path}")
    
    try:
        face_recognizer = FaceRecognitionDL(model_path=model_path)
        print("Face recognition system initialized successfully")
    except Exception as e:
        print(f"Error initializing face recognition: {str(e)}")
        return False
    
    try:
        # Initialize camera
        print("\nInitializing camera...")
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cam.isOpened():
            print("Error: Could not open camera!")
            return False
        print("Camera initialized successfully")
        
        # Set camera properties
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cam.set(cv2.CAP_PROP_FPS, 30)
        
        print("\nStarting camera test...")
        print("Press 'q' to quit\n")
        
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Error: Could not read frame!")
                break
            
            # Make a copy for drawing
            display_frame = frame.copy()
            
            try:
                # Detect faces
                face_locations = face_recognizer.get_face_locations(frame)
                
                # Draw rectangles around faces
                for (top, right, bottom, left) in face_locations:
                    # Draw rectangle
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # Show text
                    text = "Face Detected"
                    y = top - 10 if top - 10 > 10 else top + 10
                    cv2.putText(display_frame, text, (left, y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error in face detection: {str(e)}")
            
            # Show frame
            cv2.imshow('Camera Test', display_frame)
            
            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cam.release()
        cv2.destroyAllWindows()
        print("\nCamera test completed successfully")
        return True
        
    except Exception as e:
        print(f"Error in camera test: {str(e)}")
        return False

if __name__ == "__main__":
    test_camera_and_detection() 