# https://face-recognition.readthedocs.io/en/latest/face_recognition.html
# https://github.com/ageitgey/face_recognition
import cv2
import face_recognition
import threading
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
    print('Loop knowns face folder')
    for root, dirs, _ in os.walk(base_path):
    for user_folder in dirs:
        user_folder_path = os.path.join(root, user_folder)
        print(f"Processing images for user: {user_folder}")

def face_detection():
    # Load known faces and their encodings
    known_faces_encodings = []
    known_faces_names = []
    found_name = ""
    skipped_frame = 24
    frame_index = 0


    # Add known faces and their encodings
    # Replace with your known faces and their encodings

    # Access the webcam (change index if you have multiple cameras)
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, _frame0 = video_capture.read()
        # _frame = cv2.resize(_frame0, (320, 240))
        _frame = cv2.resize(_frame0, (640, 480))
        frame = cv2.flip(_frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if(frame_index >= skipped_frame):
            frame_index = 0
            # Find all face locations and encodings in the current frame
            face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample = 1)
            # faces = [(242, 407, 428, 221)]
            print(face_locations)

            if face_locations:
                top = face_locations[0][0]  # Top coordinate
                right = face_locations[0][1]  # Right coordinate
                bottom = face_locations[0][2]  # Bottom coordinate
                left = face_locations[0][3]  # Left coordinate
                # detected_face = gray_frame[y:y+h, x:x+w]
                # detected_face = gray_frame[top:bottom, left:right]
                # face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                # print(face_locations)
                # print(face_encodings)
            else:
                print('No one here')
        else:
            frame_index += 1

        # # Loop through each face found in the frame
        # for (top, right, bottom, left) in face_locations:
        #     # See if the face matches any known faces
        #     # matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)

        #     # Check if we found a match
        #     # if True in matches:
        #     #     first_match_index = matches.index(True)
        #     #     name = known_faces_names[first_match_index]

        #     # Draw a box around the face and label with the name
        #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
        if(found_name):
            cv2.putText(frame, found_name, (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()


# Function for the login form
def login_form(left_frame):
    username_label = Label(left_frame, text='Username:')
    username_label.grid(row=0, column=0)
    username_entry = Entry(left_frame)
    username_entry.grid(row=0, column=1)

    password_label = Label(left_frame, text='Password:')
    password_label.grid(row=1, column=0)
    password_entry = Entry(left_frame, show='*')
    password_entry.grid(row=1, column=1)

    login_button = Button(left_frame, text='Login')
    login_button.grid(row=2, columnspan=2)

def main():
    root = Tk()
    root.title('Combined Window')

    left_frame = Frame(root)
    left_frame.pack(side='left')

    right_frame = Frame(root)
    right_frame.pack(side='right')
    label = Label(right_frame)
    label.pack()

    camera_thread = threading.Thread(target=face_detection)

    camera_thread.start()
    
    # login_form_thread = threading.Thread(target=login_form)
    # login_form_thread.start()
    login_form(left_frame = left_frame)
    
    init_knowns_faces_thread=threading.Thread(target=init_knowns_faces)
    init_knowns_faces_thread.start()

    root.mainloop()

main()