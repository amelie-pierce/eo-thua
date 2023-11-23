import tkinter as tk
import cv2
from tkinter import ttk
import os
from tkinter import filedialog
import time
import datetime
import face_recognition
from PIL import Image, ImageTk

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

def show_camera_feed():
    global cap, is_camera_on, root, camera_label

    if is_camera_on:
        ret, frame = cap.read()
        if ret:
          image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          image = Image.fromarray(image)
          photo = ImageTk.PhotoImage(image=image)

          # Update the label with the new frame
          camera_label.config(image=photo, width=window_width // 2)
          camera_label.image = photo

          # Repeat the process after a delay (e.g., 30 milliseconds)
          root.after(30, show_camera_feed)
        else:
            # If there's an issue capturing frames, stop the camera
            toggle_camera()


# Function to start face detection
def toggle_camera():
  global is_camera_on, cap, start_button
  if not is_camera_on:
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    is_camera_on = True
    start_button.config(text="Stop Camera")
    show_camera_feed()
    
  else:
      # Stop the camera
      if cap:
        cap.release()
      is_camera_on = False
      start_button.config(text="Start Camera")

def show_form():
    global name_label, name_entry, close_button, label, register_button, scan_button
    name_label.pack(padx=250, pady=1)
    name_entry.pack(padx=250, pady=1)
    start_button.pack(padx=250, pady=1)
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

# Function to handle the "Start" button click
def on_start_click():
    label.config(text="Start Clicked!")

def update_button_states(arg):
    global start_button
    if name_entry.get():
        start_button.config(state='normal')
    else:
        start_button.config(state='disabled')

def create_frames():
  global root, start_button, name_label, name_entry, close_button, label, register_button, scan_button, camera_label, window_width

  # Create the main window
  root = tk.Tk()
  root.title("Hackathon")

  # Set the window width to half of the screen width
  window_width = 1280
  root.geometry(f"{window_width}x{root.winfo_screenheight()}")

  # Define the width of each section (50% of the screen width)
  window_width = window_width // 2

  # Create the left frame for the camera view
  left_frame = ttk.Frame(root, height=root.winfo_screenheight())
  left_frame.grid(row=0, column=0, sticky="nsew")
  # Create the label for the camera feed
  camera_label = ttk.Label(left_frame)
  camera_label.pack(expand=True, fill='both', padx=(0, window_width // 2))
  # Create the right frame for the button
  right_frame = ttk.Frame(root, height=root.winfo_screenheight())
  right_frame.grid(row=0, column=1, sticky="nsew")

  root.columnconfigure(0, weight=1)
  root.columnconfigure(1, weight=1)

  label = ttk.Label(right_frame, text="Face recognition", font=("Helvetica", 24))
  label.pack(padx=250, pady=20)
  register_button = ttk.Button(right_frame, text="Register", command=show_form)
  register_button.pack(padx=250, pady=20)
  scan_button = ttk.Button(right_frame, text="Scan", command=on_start_click)
  scan_button.pack(padx=250, pady=20)

  # Create a label and entry for name input
  name_label = ttk.Label(right_frame, text="Enter Name:")
  name_entry = ttk.Entry(right_frame)
  name_entry.bind('<KeyRelease>', update_button_states)

  # Create a button to start face detection
  start_button = ttk.Button(right_frame, text="Start Face Training", command=toggle_camera)
  start_button.config(state='disabled')
  close_button = ttk.Button(right_frame, text="Close Form", command=close_form)

  # Start the Tkinter main loop
  root.mainloop()

is_camera_on = False
cap = None
window_width = 0

create_frames()