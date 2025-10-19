import cv2

def test_camera():
    # Try different camera indices
    for camera_index in [0, 1, -1]:
        print(f"Trying camera index: {camera_index}")
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Adding CAP_DSHOW for Windows
        
        if not cap.isOpened():
            print(f"Failed to open camera with index {camera_index}")
            continue
            
        ret, frame = cap.read()
        if ret:
            print(f"Successfully accessed camera with index {camera_index}")
            cv2.imshow('Camera Test', frame)
            cv2.waitKey(2000)  # Show image for 2 seconds
            cap.release()
            cv2.destroyAllWindows()
            return camera_index
        else:
            print(f"Could not read frame from camera {camera_index}")
            cap.release()
    
    return None

if __name__ == "__main__":
    working_camera = test_camera()
    if working_camera is not None:
        print(f"\nWorking camera found at index: {working_camera}")
    else:
        print("\nNo working camera found!") 