
from tkinter import ttk
import subprocess
import cv2
import tkinter as tk
import os
from tkinter import filedialog
import time
import datetime
import face_recognition

def is_camera_stable(camera, threshold):
    # Capture two frames from the camera
    ret, frame1 = camera.read()
    ret, frame2 = camera.read()

    # Convert frames to grayscale
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between frames
    difference = cv2.absdiff(gray_frame1, gray_frame2)

    # Calculate the average difference intensity
    average_difference = cv2.mean(difference)[0]

    # Check if the average difference is below the threshold
    if average_difference < threshold:
        return True
    else:
        return False


# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a directory to store training images
training_faces_dir = 'known_faces'
if not os.path.exists(training_faces_dir):
    os.makedirs(training_faces_dir)


# Function to start face detection
def start_face_detection():
    name = name_entry.get()  # Get the name from the entry field
    count = 0
    total_images = 5
    
     # Create a folder with the user's name inside the 'known_faces' directory
    user_faces_dir = os.path.join(training_faces_dir, name)
    os.makedirs(user_faces_dir, exist_ok=True)

    # Delay time for take a photo
    last_second = -1
    delay_time_second = 2

    # Start the camera
    camera = cv2.VideoCapture(0)  # Use 0 for default webcam, change the index if using other cameras

     # Wait for the camera to stabilize
    while not is_camera_stable(camera, 10):
        time.sleep(0.1)  # Wait for a short period before checking again

    while count < total_images:
        ret, _frame = camera.read()
        if not ret:
            continue
        # Flip the frame horizontally
        frame = cv2.flip(_frame, 1)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        cur_second = datetime.datetime.now().second

        face_locations = face_recognition.face_locations(gray_frame, number_of_times_to_upsample = 1, model='hog')

        for (top, right, bottom, left) in face_locations:
            # Draw rectangles around detected faces
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 165, 0), 2)

            if(last_second < 0 or last_second >= 58):
                last_second = cur_second
            if(cur_second - last_second >= delay_time_second):
                # Save the detected face as an image with the given name
                face_img = gray_frame[top:bottom, left:right]
                count += 1
                last_second = cur_second
                face_img_path = os.path.join(user_faces_dir, f'{name}_{count}.jpg')
                cv2.imwrite(face_img_path, face_img)
                # Training one face in 1 frame
            break

        # Display the progress label on the live camera feed
        progress_text = f'Progress: {count}/{total_images}'
        cv2.putText(frame, progress_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)

        # Display the frame with rectangles around detected faces
        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()



def show_form():
    name_label.pack(padx=250, pady=1, anchor=tk.CENTER)
    name_entry.pack(padx=250, pady=1, anchor=tk.CENTER)
    start_button.pack(padx=250, pady=1, anchor=tk.CENTER)
    close_button.pack(pady=10)
    label.pack_forget()
    register_button.pack_forget()
    scan_button.pack_forget()

def close_form():
    name_label.pack_forget()
    name_entry.pack_forget()
    close_button.pack_forget()
    start_button.pack_forget()
    label.pack(padx=250, pady=20)
    register_button.pack(padx=250, pady=20)
    scan_button.pack(padx=250, pady=20)


def start_scan():
    subprocess.run(["python3", "main.py"])

def update_button_states(arg):
    if name_entry.get():
        start_button.config(state='normal')
    else:
        start_button.config(state='disabled')

root = tk.Tk()
root.title("Hackathon")

# Set the window width to 920 pixels
window_width = 1280
root.geometry(f"{window_width}x{root.winfo_screenheight()}")

# Define the width of each section (50% of the screen width)
section_width = window_width // 2

# Create the left section
left_frame = tk.Frame(root, width=section_width, height=root.winfo_screenheight(), bg="lightblue")
left_frame.grid(row=0, column=0, sticky="nsew")

# Create the right section
right_frame = tk.Frame(root, width=section_width, height=root.winfo_screenheight())
right_frame.grid(row=0, column=1, sticky="nsew")

label = tk.Label(right_frame, text="Face recognition", font=("Helvetica", 24))
label.pack(padx=250, pady=20)
register_button = tk.Button(right_frame, text="Register", command=show_form, width=15)
register_button.pack(padx=250, pady=20)
scan_button = ttk.Button(right_frame, text="Scan", command=start_scan, padding=(35, 0))
scan_button.pack(padx=250, pady=20)

# Create a label and entry for name input
name_label = tk.Label(right_frame, text="Enter Name:")

name_entry = tk.Entry(right_frame)
name_entry.bind('<KeyRelease>', update_button_states)

# Create a button to start face detection
start_button = tk.Button(right_frame, text="Start Face Training", command=start_face_detection, height=3, width=15)
start_button.config(state='disabled')

close_button = tk.Button(right_frame, text="Close Form", command=close_form)

# Start the main loop
root.mainloop()