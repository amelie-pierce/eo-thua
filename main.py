import cv2
import face_recognition
import threading
import os
from tkinter import Tk, Label, Entry, Button, Frame, messagebox
from PIL import Image, ImageTk

known_faces_dir = 'known_faces'
global user_encodings, user_names
user_encodings = []
user_names = []


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
    global user_encodings, user_names
    for root, dirs, _ in os.walk(known_faces_dir):
        for user_folder in dirs:
            user_folder_path = os.path.join(root, user_folder)
            print(f"Processing images for user: {user_folder}")
            encodings, names = recognize_faces_in_folder(user_folder_path)
            user_encodings.extend(encodings)
            user_names.extend(names)
    print(user_names)
            
# Flag to check if camera is open
camera_opened = True
count = 12

def log(text):
    current_text = log_label.cget("text")
    lines = current_text.split('\n')

    # Limit the number of lines to 10
    if len(lines) >= 10:
        lines = lines[:-1]  # Remove the oldest line

    new_text = '\n'.join([text] + lines)
    log_label.config(text=new_text)



# Function for the camera thread
def camera_thread():
    # global count
    # print(count)
    global video_capture, camera_opened
    # video_capture = cv2.VideoCapture(0)
    # print(camera_opened)

    # while camera_opened:
    #     ret, frame = video_capture.read()
    #     if ret:
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         frame = Image.fromarray(frame)
    #         frame = ImageTk.PhotoImage(image=frame)
    #         label.config(image=frame)
    #         label.image = frame
    #     # if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
    #     #     break

    # video_capture.release()
    # Load known faces and their encodings
    global user_encodings, user_names
    known_faces_encodings = []
    known_faces_names = []
    found_name = ""
    skipped_frame = 12
    frame_index = 0


    # Add known faces and their encodings
    # Replace with your known faces and their encodings

    # Access the webcam (change index if you have multiple cameras)
    video_capture = cv2.VideoCapture(0)

    while camera_opened:
        ret, _frame0 = video_capture.read()
        # _frame = cv2.resize(_frame0, (320, 240))
        _frame = cv2.resize(_frame0, (640, 480))
        frame = cv2.flip(_frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if(frame_index >= skipped_frame):
            frame_index = 0
            found_name="Unknown"
            # Find all face locations and encodings in the current frame
            face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample = 1)
            # faces = [(242, 407, 428, 221)]
            # print(face_locations)
            # log('-'.join(face_locations))
            # log("-".join(["".join(str(x)) for x in face_locations]))

            if face_locations:
                top = face_locations[0][0]  # Top coordinate
                right = face_locations[0][1]  # Right coordinate
                bottom = face_locations[0][2]  # Bottom coordinate
                left = face_locations[0][3]  # Left coordinate
                # detected_face = gray_frame[y:y+h, x:x+w]
                # detected_face = gray_frame[top:bottom, left:right]
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                # print(face_encodings)
                
                if(face_encodings):
                    face_encoding = face_encodings[0]
                    # Loop through each face found in the frame
                    for (top, right, bottom, left) in face_locations:
                        # See if the face matches any known faces
                        matches = face_recognition.compare_faces(user_encodings, face_encoding, tolerance=0.4)

                        # Check if we found a match
                        if True in matches:
                            first_match_index = matches.index(True)
                            found_name = user_names[first_match_index]
                            break
                log(text = found_name)
            else:
                log(text = 'No one here')
        else:
            frame_index += 1
            
        
        # Display the resulting frame
        # cv2.imshow('Video', frame)
        frame = Image.fromarray(rgb_frame)
        frame = ImageTk.PhotoImage(image=frame)
        label.config(image=frame)
        label.image = frame
        # log(text=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

# Function to handle login button click
def login_clicked():
    username = username_entry.get()
    password = password_entry.get()

    # Your authentication logic goes here (replace this with your actual login verification)
    if username == "user" and password == "password":
        messagebox.showinfo("Login", "Login Successful")
    else:
        messagebox.showerror("Login", "Invalid username or password")

def _quit():
    root.quit()     
    root.destroy()

# Function to close the camera stream and window
def close_camera():
    global count
    count +=1
    global camera_opened, video_capture
    camera_opened = False
    
    if 'video_capture' in globals():
        video_capture.release()
    _quit()

# Function to confirm exit with "Ctrl + q"
def confirm_exit(event):
    if event.keysym == 'q' and event.state == 4:  # Checking for 'q' and Ctrl key combination
        confirm = messagebox.askokcancel("Confirm Exit", "Are you sure you want to exit?")
        if confirm:
            close_camera()

# Init known faces
init_knowns_faces()
# init_knowns_faces_thread=threading.Thread(target=init_knowns_faces)
# init_knowns_faces_thread.start()

# Create main window
root = Tk()
root.title('Login & Camera Stream')

# Make the window full screen
root.attributes('-fullscreen', True)

# Create frames for left and right sections
left_frame = Frame(root)
left_frame.pack(side='left', fill='both', expand=True)

right_frame = Frame(root)
right_frame.pack(side='right', fill='both', expand=True)

# Label in the left frame for displaying 'Login'
hello_label = Label(left_frame, text='Login', font=('Arial', 20))
hello_label.grid(row=0, column=0, columnspan=2, pady=10)

# Login form components
username_label = Label(left_frame, text='Username:')
username_label.grid(row=1, column=0)
username_entry = Entry(left_frame)
username_entry.grid(row=1, column=1)

password_label = Label(left_frame, text='Password:')
password_label.grid(row=2, column=0)
password_entry = Entry(left_frame, show='*')
password_entry.grid(row=2, column=1)

login_button = Button(left_frame, text='Login', command=login_clicked)
login_button.grid(row=3, columnspan=2, pady=10)

# # Label in the right frame for displaying camera stream
label = Label(right_frame)
label.pack(fill='both', expand=True)

log_label = Label(left_frame, text='Logs...')
log_label.grid(row=4, columnspan=2,  pady=10 )


# Start camera thread
camera_thread = threading.Thread(target=camera_thread)
camera_thread.start()

# Bind closing event to the window
root.protocol("WM_DELETE_WINDOW", close_camera)

# Bind Ctrl + q to confirm exit
root.bind_all("<Control-q>", confirm_exit)

root.mainloop()

