import tkinter as tk
from tkinter import *
import os, cv2
import shutil
import csv
import numpy as np
from PIL import ImageTk, Image
import pandas as pd
import datetime
import time
import tkinter.font as font
import pyttsx3
import tkinter.ttk as ttk
import tkinter.messagebox as messagebox

# project module
import show_attendance
import takeImage
import trainImage
import automaticAttedance

# engine = pyttsx3.init()
# engine.say("Welcome!")
# engine.say("Please browse through your options..")
# engine.runAndWait()


def text_to_speech(user_text):
    engine = pyttsx3.init()
    engine.say(user_text)
    engine.runAndWait()


# Create required directories
model_path = "models"
trainimage_path = "TrainingImage"
studentdetail_path = "StudentDetails/studentdetails.csv"
attendance_path = "Attendance"

# Default subjects list
SUBJECTS = [
    "CSE-A", "CSE-B", "CSE-C",
    "IT-A", "IT-B",
    "ECE-A", "ECE-B",
    "EEE-A", "EEE-B"
]

# Create directories and subject folders
for path in [model_path, trainimage_path, "StudentDetails", attendance_path]:
    os.makedirs(path, exist_ok=True)
    
# Create subject directories in Attendance folder
for subject in SUBJECTS:
    os.makedirs(os.path.join(attendance_path, subject), exist_ok=True)

# Create student details file if it doesn't exist
if not os.path.exists(studentdetail_path):
    with open(studentdetail_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Enrollment', 'Name'])

window = Tk()
window.title("Face Recognition Attendance System")
window.geometry("1280x720")
dialog_title = "QUIT"
dialog_text = "Are you sure want to close?"
window.configure(background="#1c1c1c")  # Dark theme

# Global variable for selected subject
current_subject = None

# Subject selection frame
def create_subject_frame():
    global current_subject
    subject_frame = Frame(window, bg="#1c1c1c", relief=RIDGE, bd=5)
    subject_frame.place(x=400, y=150, width=480, height=100)
    
    # Subject label
    Label(
        subject_frame,
        text="Current Subject:",
        bg="#1c1c1c",
        fg="yellow",
        font=("Verdana", 16, "bold")
    ).pack(pady=5)
    
    # Subject selection
    subject_var = StringVar()
    subjects = [
        "CSE-A", "CSE-B", "CSE-C",
        "IT-A", "IT-B",
        "ECE-A", "ECE-B",
        "EEE-A", "EEE-B"
    ]
    
    # Create and configure style for combobox
    style = ttk.Style()
    try:
        style.theme_use('custom')
    except:
        try:
            style.theme_create('custom', parent='alt', settings={
                'TCombobox': {
                    'configure': {
                        'selectbackground': '#333333',
                        'fieldbackground': '#333333',
                        'background': '#333333',
                        'foreground': 'yellow'
                    }
                }
            })
            style.theme_use('custom')
        except:
            pass

    # Configure the combobox colors
    window.option_add('*TCombobox*Listbox.background', '#333333')
    window.option_add('*TCombobox*Listbox.foreground', 'yellow')
    window.option_add('*TCombobox*Listbox.selectBackground', '#444444')
    window.option_add('*TCombobox*Listbox.selectForeground', 'yellow')
    
    def on_subject_change(event):
        global current_subject
        current_subject = subject_var.get()
        print(f"Selected subject: {current_subject}")
        
    # Subject dropdown
    subject_combo = ttk.Combobox(
        subject_frame,
        textvariable=subject_var,
        values=subjects,
        state="readonly",
        font=("Verdana", 14),
        width=25
    )
    subject_combo.pack(pady=5)
    subject_combo.set("Select Subject")  # Default text
    subject_combo.bind('<<ComboboxSelected>>', on_subject_change)

# Create the subject selection frame
create_subject_frame()

# to destroy screen
def del_sc1():
    sc1.destroy()


# error message for name and no
def err_screen():
    global sc1
    sc1 = tk.Tk()
    sc1.geometry("400x110")
    sc1.iconbitmap("AMS.ico")
    sc1.title("Warning!!")
    sc1.configure(background="#1c1c1c")
    sc1.resizable(0, 0)
    tk.Label(
        sc1,
        text="Enrollment & Name required!!!",
        fg="yellow",
        bg="#1c1c1c",  # Dark background for the error window
        font=("Verdana", 16, "bold"),
    ).pack()
    tk.Button(
        sc1,
        text="OK",
        command=del_sc1,
        fg="yellow",
        bg="#333333",  # Darker button color
        width=9,
        height=1,
        activebackground="red",
        font=("Verdana", 16, "bold"),
    ).place(x=110, y=50)

def testVal(inStr, acttyp):
    if acttyp == "1":  # insert
        if not inStr.isdigit():
            return False
    return True


# Title and logo
titl = tk.Label(window, bg="#1c1c1c", relief=RIDGE, bd=10, font=("Verdana", 30, "bold"))
titl.pack(fill=X)

# Logo
logo = Image.open("UI_Image/0001.png")
logo = logo.resize((50, 47), Image.LANCZOS)
logo1 = ImageTk.PhotoImage(logo)
l1 = tk.Label(window, image=logo1, bg="#1c1c1c")
l1.place(x=470, y=10)

# Main title
titl = tk.Label(
    window, 
    text="Face Recognition Attendance System", 
    bg="#1c1c1c", 
    fg="yellow", 
    font=("Verdana", 27, "bold")
)
titl.place(x=525, y=12)

# Welcome text
a = tk.Label(
    window,
    text="Welcome to Smart Attendance System",
    bg="#1c1c1c",
    fg="yellow",
    bd=10,
    font=("Verdana", 35, "bold"),
)
a.pack(pady=(0, 20))

# Load and place images
ri = Image.open("UI_Image/register.png")
r = ImageTk.PhotoImage(ri)
label1 = Label(window, image=r, bg="#1c1c1c")
label1.image = r
label1.place(x=100, y=270)

ai = Image.open("UI_Image/attendance.png")
a = ImageTk.PhotoImage(ai)
label2 = Label(window, image=a, bg="#1c1c1c")
label2.image = a
label2.place(x=980, y=270)

vi = Image.open("UI_Image/verifyy.png")
v = ImageTk.PhotoImage(vi)
label3 = Label(window, image=v, bg="#1c1c1c")
label3.image = v
label3.place(x=600, y=270)

def TakeImageUI():
    ImageUI = Tk()
    ImageUI.title("Register New Student")
    ImageUI.geometry("780x480")
    ImageUI.configure(background="#1c1c1c")  # Dark background for the image window
    ImageUI.resizable(0, 0)
    titl = tk.Label(ImageUI, bg="#1c1c1c", relief=RIDGE, bd=10, font=("Verdana", 30, "bold"))
    titl.pack(fill=X)
    # image and title
    titl = tk.Label(
        ImageUI, text="Register Your Face", bg="#1c1c1c", fg="green", font=("Verdana", 30, "bold"),
    )
    titl.place(x=270, y=12)

    # heading
    a = tk.Label(
        ImageUI,
        text="Enter the details",
        bg="#1c1c1c",  # Dark background for the details label
        fg="yellow",  # Bright yellow text color
        bd=10,
        font=("Verdana", 24, "bold"),
    )
    a.place(x=280, y=75)

    # ER no
    lbl1 = tk.Label(
        ImageUI,
        text="Enrollment No",
        width=10,
        height=2,
        bg="#1c1c1c",
        fg="yellow",
        bd=5,
        relief=RIDGE,
        font=("Verdana", 14),
    )
    lbl1.place(x=120, y=130)
    txt1 = tk.Entry(
        ImageUI,
        width=17,
        bd=5,
        validate="key",
        bg="#333333",  # Dark input background
        fg="yellow",  # Bright text color for input
        relief=RIDGE,
        font=("Verdana", 18, "bold"),
    )
    txt1.place(x=250, y=130)
    txt1["validatecommand"] = (txt1.register(testVal), "%P", "%d")

    # name
    lbl2 = tk.Label(
        ImageUI,
        text="Name",
        width=10,
        height=2,
        bg="#1c1c1c",
        fg="yellow",
        bd=5,
        relief=RIDGE,
        font=("Verdana", 14),
    )
    lbl2.place(x=120, y=200)
    txt2 = tk.Entry(
        ImageUI,
        width=17,
        bd=5,
        bg="#333333",  # Dark input background
        fg="yellow",  # Bright text color for input
        relief=RIDGE,
        font=("Verdana", 18, "bold"),
    )
    txt2.place(x=250, y=200)

    lbl3 = tk.Label(
        ImageUI,
        text="Status",
        width=10,
        height=2,
        bg="#1c1c1c",
        fg="yellow",
        bd=5,
        relief=RIDGE,
        font=("Verdana", 14),
    )
    lbl3.place(x=120, y=270)

    message = tk.Label(
        ImageUI,
        text="",
        width=32,
        height=2,
        bd=5,
        bg="#333333",  # Dark background for messages
        fg="yellow",  # Bright text color for messages
        relief=RIDGE,
        font=("Verdana", 14, "bold"),
    )
    message.place(x=250, y=270)

    def take_image():
        l1 = txt1.get()
        l2 = txt2.get()
        takeImage.TakeImage(
            l1,
            l2,
            trainimage_path,
            message,
            err_screen,
            text_to_speech,
        )
        txt1.delete(0, "end")
        txt2.delete(0, "end")

    # take Image button
    # image
    takeImg = tk.Button(
        ImageUI,
        text="Take Images",
        command=take_image,
        bd=10,
        font=("Verdana", 18, "bold"),
        bg="#333333",  # Dark background for the button
        fg="yellow",  # Bright text color for the button
        height=2,
        width=12,
        relief=RIDGE,
    )
    takeImg.place(x=130, y=350)

    def train_image():
        trainImage.TrainImage(
            trainimage_path,
            model_path,
            message,
            text_to_speech,
        )

    # train Image function call
    trainImg = tk.Button(
        ImageUI,
        text="Train Model",
        command=train_image,
        bd=10,
        font=("Verdana", 18, "bold"),
        bg="#333333",  # Dark background for the button
        fg="yellow",  # Bright text color for the button
        height=2,
        width=12,
        relief=RIDGE,
    )
    trainImg.place(x=360, y=350)


def automatic_attendance():
    global current_subject
    if not current_subject:
        messagebox.showwarning("Warning", "Please select a subject first!")
        text_to_speech("Please select a subject first")
        return
    
    # Create subject directory if it doesn't exist
    subject_dir = os.path.join("Attendance", current_subject)
    os.makedirs(subject_dir, exist_ok=True)
    
    # Start attendance
    print(f"\nStarting attendance for {current_subject}")
    automaticAttedance.take_attendance_with_recognition(current_subject)

def view_attendance():
    show_attendance.subjectchoose(text_to_speech)

# Action buttons
register_btn = tk.Button(
    window,
    text="Register New Student",
    command=TakeImageUI,
    bd=10,
    font=("Verdana", 16),
    bg="black",
    fg="yellow",
    height=2,
    width=17,
)
register_btn.place(x=100, y=520)

attendance_btn = tk.Button(
    window,
    text="Take Attendance",
    command=automatic_attendance,
    bd=10,
    font=("Verdana", 16),
    bg="black",
    fg="yellow",
    height=2,
    width=17,
)
attendance_btn.place(x=600, y=520)

view_btn = tk.Button(
    window,
    text="View Attendance",
    command=view_attendance,
    bd=10,
    font=("Verdana", 16),
    bg="black",
    fg="yellow",
    height=2,
    width=17,
)
view_btn.place(x=1000, y=520)

exit_btn = tk.Button(
    window,
    text="EXIT",
    bd=10,
    command=quit,
    font=("Verdana", 16),
    bg="black",
    fg="yellow",
    height=2,
    width=17,
)
exit_btn.place(x=600, y=660)

window.mainloop()
