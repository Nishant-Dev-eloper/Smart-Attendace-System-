import csv
import os, cv2
import numpy as np
import pandas as pd
import datetime
import time
from PIL import ImageTk, Image
from face_recognition_dl import FaceRecognitionDL


# Train Image
def TrainImage(trainimage_path, model_path, message, text_to_speech):
    try:
        print("\nInitializing face recognition system...")
        # Initialize face recognition system
        os.makedirs(model_path, exist_ok=True)
        face_recognizer = FaceRecognitionDL(model_path=model_path)
        print("Face recognition system initialized")
        
        # Get all subdirectories (one per person)
        print("\nScanning for training images...")
        image_dirs = [os.path.join(trainimage_path, d) for d in os.listdir(trainimage_path)]
        if not image_dirs:
            error_msg = "No training images found! Please register faces first."
            print(error_msg)
            if message:
                message.configure(text=error_msg)
            if text_to_speech:
                text_to_speech(error_msg)
            return False
            
        total_faces = 0
        processed_people = 0
        failed_images = []
        
        # Process each person's directory
        for directory in image_dirs:
            try:
                # Get enrollment and name from directory name
                dir_name = os.path.basename(directory)
                enrollment, name = dir_name.split('_', 1)
                print(f"\nProcessing images for {name} (Enrollment: {enrollment})")
                
                # Process each image in the directory
                dir_faces = 0
                dir_failures = 0
                for img_file in os.listdir(directory):
                    if img_file.endswith(('.jpg', '.png')):
                        img_path = os.path.join(directory, img_file)
                        
                        try:
                            # Read and convert image
                            image = cv2.imread(img_path)
                            if image is None:
                                raise ValueError(f"Could not read image: {img_path}")
                                
                            # Add face to recognition system
                            if face_recognizer.add_face(name, enrollment, image):
                                total_faces += 1
                                dir_faces += 1
                                print(f"Processed {dir_faces} faces for {name} ({dir_failures} failures)...", end='\r')
                                
                                if message:
                                    message.configure(text=f"Training: {total_faces} faces processed...")
                            else:
                                dir_failures += 1
                                failed_images.append(img_path)
                                
                        except Exception as e:
                            dir_failures += 1
                            failed_images.append(f"{img_path}: {str(e)}")
                            continue
                
                if dir_faces > 0:
                    processed_people += 1
                    print(f"\nSuccessfully processed {dir_faces} faces for {name} ({dir_failures} failures)")
                else:
                    print(f"\nNo valid faces found in images for {name} ({dir_failures} failures)")
                            
            except Exception as e:
                print(f"\nError processing directory {directory}: {str(e)}")
                continue
                
        if total_faces > 0:
            # Save the updated embeddings
            face_recognizer.save_embeddings()
            
            success_msg = f"Training completed successfully! Processed {total_faces} faces from {processed_people} people."
            if failed_images:
                success_msg += f"\nWarning: {len(failed_images)} images failed processing."
                print("\nFailed images:")
                for fail in failed_images[:10]:  # Show first 10 failures
                    print(f"- {fail}")
                if len(failed_images) > 10:
                    print(f"... and {len(failed_images) - 10} more")
                    
            print(f"\n{success_msg}")
            if message:
                message.configure(text=success_msg)
            if text_to_speech:
                text_to_speech(success_msg)
            return True
        else:
            error_msg = "No valid faces were found in the training images."
            print(f"\n{error_msg}")
            if message:
                message.configure(text=error_msg)
            if text_to_speech:
                text_to_speech(error_msg)
            return False
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        print(error_msg)
        if message:
            message.configure(text=error_msg)
        if text_to_speech:
            text_to_speech(error_msg)
        return False


def getImagesAndLables(path):
    print("Loading training images...")
    faces = []
    Ids = []
    
    # Get all subdirectories
    try:
        newdir = [os.path.join(path, d) for d in os.listdir(path)]
        if not newdir:
            print("No training images found!")
            return faces, Ids
            
        # Get all image paths
        imagePaths = []
        for directory in newdir:
            try:
                for f in os.listdir(directory):
                    if f.endswith('.jpg') or f.endswith('.png'):
                        imagePaths.append(os.path.join(directory, f))
            except Exception as e:
                print(f"Error reading directory {directory}: {str(e)}")
                continue
        
        total_images = len(imagePaths)
        print(f"Found {total_images} images for training")
        
        # Process each image
        for i, imagePath in enumerate(imagePaths, 1):
            try:
                print(f"Processing image {i}/{total_images}: {os.path.basename(imagePath)}", end='\r')
                pilImage = Image.open(imagePath).convert("L")
                imageNp = np.array(pilImage, "uint8")
                Id = int(os.path.split(imagePath)[-1].split("_")[1])
                faces.append(imageNp)
                Ids.append(Id)
            except Exception as e:
                print(f"\nError processing image {imagePath}: {str(e)}")
                continue
                
        print("\nImage processing completed!")
        return faces, Ids
        
    except Exception as e:
        print(f"Error reading training directory: {str(e)}")
        return faces, Ids
