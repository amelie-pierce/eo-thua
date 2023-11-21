import cv2
import face_recognition
import threading
import os
from tkinter import Tk, Label, Entry, Button, Frame
from PIL import Image, ImageTk

known_faces_dir = 'known_faces'

# Function to load images and perform face recognition
def recognize_faces_in_folder(folder_path):
    known_encodings = []
    known_names = []

    # Load encodings and names from images in user folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.jpg', '.jpeg', '.png', '.gif')):  # Check for image files
            image_path = os.path.join(folder_path, file_name)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)

            if encoding:
                known_encodings.append(encoding[0])
                known_names.append(os.path.basename(folder_path))

    return known_encodings, known_names

def init_knowns_faces():
    print('Loop knowns face folder...')
    for root, dirs, _ in os.walk(known_faces_dir):
        for user_folder in dirs:
            user_folder_path = os.path.join(root, user_folder)
            print(f"Processing images for user: {user_folder}")
            user_encodings, user_names = recognize_faces_in_folder(user_folder_path)
            print(user_encodings, user_names)


init_knowns_faces()
