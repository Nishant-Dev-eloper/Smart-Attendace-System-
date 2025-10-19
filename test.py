import requests
import cv2
import numpy as np
from requests.exceptions import RequestException
import sys

# IP camera URL
url = "http://192.168.0.6:8080/shot.jpg"

def check_camera_connection():
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except RequestException:
        return False

def main():
    # Check if camera is accessible
    if not check_camera_connection():
        print("Error: Cannot connect to IP camera. Please check the URL and connection.")
        sys.exit(1)

    try:
        while True:
            try:
                # Get image from IP camera
                cam = requests.get(url, timeout=2)
                imgNp = np.array(bytearray(cam.content), dtype=np.uint8)
                img = cv2.imdecode(imgNp, -1)
                
                if img is None:
                    print("Error: Failed to decode image")
                    continue
                
                cv2.imshow("cam", img)

                # Break loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                    
            except RequestException as e:
                print(f"Network error: {e}")
                continue
            except Exception as e:
                print(f"Error: {e}")
                break
                
    finally:
        # Cleanup
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
