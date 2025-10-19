import tkinter as tk
from tkinter import *
import os
import cv2
import csv
import numpy as np
from PIL import ImageTk, Image
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
from face_recognition_dl import FaceRecognitionDL

# Create required directories if they don't exist
required_dirs = [
    "TrainingImage",
    "TrainingImageLabel",
    "StudentDetails",
    "Attendance"
]

for dir_path in required_dirs:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

# Create StudentDetails.csv if it doesn't exist
studentdetail_path = os.path.join("StudentDetails", "studentdetails.csv")
if not os.path.exists(studentdetail_path):
    with open(studentdetail_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Enrollment", "Name"])
        print("Created StudentDetails.csv")

# Default subjects list
SUBJECTS = [
    "CSE-A", "CSE-B", "CSE-C",
    "IT-A", "IT-B",
    "ECE-A", "ECE-B",
    "EEE-A", "EEE-B"
]

def take_attendance_with_recognition(subject):
    try:
        print("\nInitializing face recognition system...")
        # Initialize face recognition system with model path
        model_path = os.path.join(os.getcwd(), "models")
        os.makedirs(model_path, exist_ok=True)
        print(f"Using model path: {model_path}")
        
        face_recognizer = FaceRecognitionDL(model_path=model_path)
        print("Face recognition system initialized")
        
        # Check if we have any trained faces
        if not face_recognizer.embeddings_db:
            error_msg = "No trained faces found! Please register and train faces first."
            print(error_msg)
            return
            
        print(f"Loaded {len(face_recognizer.embeddings_db)} trained faces")
        
        # Initialize camera
        print("\nInitializing camera...")
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cam.isOpened():
            print("Error: Could not open camera!")
            return
        print("Camera initialized successfully")
            
        # Set camera properties
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cam.set(cv2.CAP_PROP_FPS, 30)
        
        # Create attendance directory if it doesn't exist
        attendance_dir = "Attendance"
        os.makedirs(attendance_dir, exist_ok=True)
        
        # Create attendance file
        date = datetime.datetime.now().strftime("%Y-%m-%d")
        subject_dir = os.path.join(attendance_dir, subject)
        os.makedirs(subject_dir, exist_ok=True)
        attendance_file = os.path.join(subject_dir, f"{date}.csv")
        print(f"\nAttendance will be saved to: {attendance_file}")
        
        # Initialize attendance record
        attendance_record = set()
        
        print("\nStarting attendance capture...")
        print("Press 'q' to quit\n")
        
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Error: Could not read frame!")
                break
                
            # Make a copy for drawing
            display_frame = frame.copy()
            
            # Detect faces
            face_locations = face_recognizer.get_face_locations(frame)
            
            for (top, right, bottom, left) in face_locations:
                # Extract face region
                face_img = frame[top:bottom, left:right]
                
                # Identify face
                identity, confidence = face_recognizer.identify_face(face_img)
                
                # Draw rectangle and put text
                if identity != "Unknown":
                    color = (0, 255, 0)  # Green for recognized
                    enrollment, name = identity.split('_', 1)
                    text = f"{name} ({confidence:.2f})"
                    
                    # Add to attendance if not already present
                    if identity not in attendance_record:
                        attendance_record.add(identity)
                        # Write to CSV
                        with open(attendance_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            if os.path.getsize(attendance_file) == 0:
                                writer.writerow(['Enrollment', 'Name', 'Time'])
                            time_str = datetime.datetime.now().strftime("%H:%M:%S")
                            writer.writerow([enrollment, name, time_str])
                        print(f"Marked attendance for: {name}")
                else:
                    color = (0, 0, 255)  # Red for unknown
                    text = f"Unknown ({confidence:.2f})"
                
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                y = top - 10 if top - 10 > 10 else top + 10
                cv2.putText(display_frame, text, (left, y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            
            # Show frame
            cv2.imshow('Attendance', display_frame)
            
            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cam.release()
        cv2.destroyAllWindows()
        
        print(f"\nAttendance saved to: {attendance_file}")
        print(f"Total attendance marked: {len(attendance_record)}")
        
    except Exception as e:
        print(f"Error taking attendance: {str(e)}")
        import traceback
        traceback.print_exc()

def subjectChoose(text_to_speech):
    class SubjectSelector:
        def __init__(self):
            self.selected_subject = None
            self.root = tk.Tk()
            self.setup_window()
            
        def setup_window(self):
            self.root.title("Subject Selection")
            self.root.geometry("580x320")
            self.root.configure(background="#1c1c1c")
            self.root.resizable(0, 0)
            
            # Center the window
            window_width = 580
            window_height = 320
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            center_x = int(screen_width/2 - window_width/2)
            center_y = int(screen_height/2 - window_height/2)
            self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
            
            # Main frame
            main_frame = tk.Frame(self.root, bg="#1c1c1c")
            main_frame.pack(expand=True, fill='both', padx=20, pady=20)
            
            # Title
            title_label = tk.Label(
                main_frame,
                text="Select Subject",
                bg="#1c1c1c",
                fg="yellow",
                font=("Verdana", 24, "bold")
            )
            title_label.pack(pady=(0, 30))
            
            # Subject selection
            self.subject_var = tk.StringVar()
            
            # Create and configure style for combobox
            style = ttk.Style()
            try:
                # Try to use existing theme if it exists
                style.theme_use('custom')
            except tk.TclError:
                # Create theme only if it doesn't exist
                style.theme_create('custom', parent='alt', settings={
                    'TCombobox': {
                        'configure': {
                            'selectbackground': '#333333',
                            'fieldbackground': '#333333',
                            'background': '#333333',
                            'foreground': 'yellow',
                            'arrowcolor': 'yellow'
                        }
                    }
                })
                style.theme_use('custom')

            # Configure the combobox colors directly
            self.root.option_add('*TCombobox*Listbox.background', '#333333')
            self.root.option_add('*TCombobox*Listbox.foreground', 'yellow')
            self.root.option_add('*TCombobox*Listbox.selectBackground', '#444444')
            self.root.option_add('*TCombobox*Listbox.selectForeground', 'yellow')
            
            # Combobox
            self.combo = ttk.Combobox(
                main_frame,
                textvariable=self.subject_var,
                values=SUBJECTS,
                state="readonly",
                font=("Verdana", 16),
                width=30
            )
            self.combo.pack(pady=10)
            self.combo.set(SUBJECTS[0])  # Set default value
            
            # Start button
            start_button = tk.Button(
                main_frame,
                text="Start Attendance",
                command=self.on_select,
                bd=3,
                font=("Verdana", 16, "bold"),
                bg="#333333",
                fg="yellow",
                activebackground="#444444",
                activeforeground="yellow",
                relief=tk.RAISED,
                width=20,
                cursor="hand2"
            )
            start_button.pack(pady=20)
            
            # Status label
            self.status_label = tk.Label(
                main_frame,
                text="Choose a subject and press Enter or click Start",
                bg="#1c1c1c",
                fg="yellow",
                font=("Verdana", 12)
            )
            self.status_label.pack(pady=10)
            
            # Bind keyboard shortcuts
            self.root.bind('<Return>', lambda e: self.on_select())
            self.root.bind('<Escape>', lambda e: self.on_cancel())
            
            # Handle window close button
            self.root.protocol("WM_DELETE_WINDOW", self.on_cancel)
            
            # Focus on combobox
            self.combo.focus_set()
            
        def on_select(self):
            selected = self.subject_var.get()
            if selected:
                self.selected_subject = selected
                self.root.quit()
            else:
                self.status_label.config(text="Please choose a subject")
                if text_to_speech:
                    text_to_speech("Please choose a subject")
                    
        def on_cancel(self):
            self.selected_subject = None
            self.root.quit()
            
        def run(self):
            self.root.mainloop()
            self.root.destroy()
            return self.selected_subject
    
    try:
        # Create and run the subject selector
        selector = SubjectSelector()
        selected_subject = selector.run()
        
        # If a subject was selected, start attendance
        if selected_subject:
            # Create subject directory if it doesn't exist
            subject_dir = os.path.join("Attendance", selected_subject)
            os.makedirs(subject_dir, exist_ok=True)
            
            print(f"\nStarting attendance for {selected_subject}")
            take_attendance_with_recognition(selected_subject)
        else:
            print("\nAttendance cancelled")
            
    except Exception as e:
        print(f"Error in subject selection: {str(e)}")
        if text_to_speech:
            text_to_speech("Error in subject selection")
    finally:
        # Ensure any remaining windows are cleaned up
        try:
            selector.root.destroy()
        except:
            pass
