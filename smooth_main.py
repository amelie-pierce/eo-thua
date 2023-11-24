import cv2
import face_recognition
import threading
import os
from tkinter import Tk, Label, Entry, Button, Frame, messagebox
from PIL import Image, ImageTk
import datetime


import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
from mediapipe.python.solutions import drawing_utils as mp_drawing
from media_pipe_utils import _normalized_to_pixel_coordinates

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')


known_faces_dir = 'known_faces'
global user_encodings, user_names
user_encodings = []
user_names = []
camera_opened = True
global ready 
ready = False
# user detected count
user_count = {}
# detected detected_times for make sure right person
detected_times = 2
skipped_frame = 4
# less tolerance => high accuracy compare faces
tolerance=0.4

# for live camera
global rgb_frame, live_time, detection_tolerance, video_capture
detection_tolerance = 0.9
rgb_frame = None
video_capture = None

####

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
    global user_encodings, user_names, ready
    for root, dirs, _ in os.walk(known_faces_dir):
        for user_folder in dirs:
            user_folder_path = os.path.join(root, user_folder)
            print(f"Processing images for user: {user_folder}")
            encodings, names = recognize_faces_in_folder(user_folder_path)
            user_encodings.extend(encodings)
            user_names.extend(names)
    ready = True
    print(user_names)
    log(text="Loaded known_faces")
    
    
# init_knowns_faces()
init_knowns_faces_thread=threading.Thread(target=init_knowns_faces)
init_knowns_faces_thread.start()

def get_current_time():
    return datetime.datetime.now().strftime("%Y/%m/%d-%H:%M:%S")

def update_detected_label(text):
    current_text = detected_label.cget("text")
    lines = current_text.split('\n')

    # Limit the number of lines to 10
    if len(lines) >= 10:
        lines = lines[:-1]  # Remove the oldest line

    new_text = '\n'.join([text] + lines)
    detected_label.config(text=new_text)
    

def log(text):
    current_text = log_label.cget("text")
    lines = current_text.split('\n')

    # Limit the number of lines to 10
    if len(lines) >= 10:
        lines = lines[:-1]  # Remove the oldest line

    new_text = '\n'.join([text] + lines)
    log_label.config(text=new_text)


def smooth_camera():
    global rgb_frame, camera_opened
    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=detection_tolerance) as face_detection:
        video_capture = cv2.VideoCapture(0)
        
        while camera_opened:
            ret, _frame0 = video_capture.read()
            if not ret:
                continue
            if(not ready):
                continue
            # _frame = cv2.resize(_frame0, (320, 240))
            frame = cv2.resize(_frame0, (640, 480))
            # frame = cv2.flip(_frame0, 1)
            frame.flags.writeable = False
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
           
            results = face_detection.process(rgb_frame)
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(rgb_frame, detection)
                   
            frame = Image.fromarray(rgb_frame)
            frame = ImageTk.PhotoImage(image=frame)
            label.config(image=frame)
            label.image = frame

# Function for the camera thread
def camera_thread():
    global video_capture, user_encodings, user_names, ready, camera_opened, skipped_frame, tolerance, live_time, rgb_frame
    # video_capture = cv2.VideoCapture(0)

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
    # global face_detection_encodings
    known_faces_encodings = []
    known_faces_names = []
    found_name = ""
    frame_index = 0


    # Add known faces and their encodings
    # Replace with your known faces and their encodings

    # Access the webcam (change index if you have multiple cameras)
    # video_capture = cv2.VideoCapture(0)

    while camera_opened:
        live_time = get_current_time()
        # ret, _frame0 = video_capture.read()
        # if not ret:
        #     continue
        # # _frame = cv2.resize(_frame0, (320, 240))
        # # _frame = cv2.resize(_frame0, (640, 480))
        # frame = cv2.flip(_frame0, 1)
        # frame.flags.writeable = False
        # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if rgb_frame is None:
                continue
        
        # put time
        # cv2.putText(rgb_frame,live_time , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
    
        if(not ready):
            # frame = Image.fromarray(rgb_frame)
            # frame = ImageTk.PhotoImage(image=frame)
            # label.config(image=frame)
            # label.image = frame
            continue
        
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # , model='hog'
        face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample = 1, model='hog')
        # for (top, right, bottom, left) in face_locations:
        #         cv2.rectangle(rgb_frame, (left, top), (right, bottom), (255, 165, 0), 2)
                # break

       

        if(frame_index >= skipped_frame):
            # face_detection_encodings = []
            frame_index = 0
            found_name="Unknown"
            # Find all face locations and encodings in the current frame
            # faces = [(242, 407, 428, 221)]
            # print(face_locations)
            # log('-'.join(face_locations))
            # log("-".join(["".join(str(x)) for x in face_locations]))
            
            
            if face_locations:
                # top = face_locations[0][0]  # Top coordinate
                # right = face_locations[0][1]  # Right coordinate
                # bottom = face_locations[0][2]  # Bottom coordinate
                # left = face_locations[0][3]  # Left coordinate
                # detected_face = gray_frame[y:y+h, x:x+w]
                # detected_face = gray_frame[top:bottom, left:right]
                # facesv1 = [item[0] for item in faces]
                # [face_locations[0]]
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                # face_detection_encodings = face_encodings
                # print(face_encodings)
                
                if(face_encodings):
                    # face_encoding = face_encodings[0]
                    for(face_encoding) in face_encodings:
                        found_name = 'Unknown'
                        # Loop through each face found in the frame
                        # for (top, right, bottom, left) in face_locations:
                        # See if the face matches any known faces
                        matches = face_recognition.compare_faces(user_encodings, face_encoding, tolerance=tolerance)

                        # Check if we found a match
                        if True in matches:
                            first_match_index = matches.index(True)
                            found_name = user_names[first_match_index]
                            # cv2.putText(rgb_frame,found_name , (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                        
                        log(text = live_time + ':  '+found_name) 
                        on_found_name(found_name)
                        # TODO: implement found name here
                #             # For case one person/screen 
                #             # Check if we found a match
                #             if True in matches:
                #                 first_match_index = matches.index(True)
                #                 found_name = user_names[first_match_index]
                #                 # draw name on camera
                #                 cv2.putText(rgb_frame,found_name , (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                #                 # break
                # log(text = live_time + ':  '+found_name)
            else:
                log(text = 'No one here')
        else:
            frame_index += 1
            
        
        # # Display the resulting frame
        # # cv2.imshow('Video', frame)
        # frame = Image.fromarray(rgb_frame)
        # frame = ImageTk.PhotoImage(image=frame)
        # label.config(image=frame)
        # label.image = frame
        # # log(text=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the OpenCV windows
    if video_capture is not None:
        video_capture.release()
    cv2.destroyAllWindows()

def update_count(found_name):
    user_count[found_name] = user_count.get(found_name, 0) + 1

def on_found_name(found_name):
    if(found_name == 'Unknown'):
        count_unknown=1
    else:
        update_count(found_name)
        if (user_count[found_name] == detected_times):
            print(f"User: {found_name} = {detected_times} times")
            update_detected_label(get_current_time() + ': ' + found_name)
            # TODO: A Dũng chỗ này xử lý popup success hiện thông user lên được nè (multiple face/frame luôn nha)
            # sau đó click reset lại user_count[found_name] thì mới vào đây lần nữa
            # => đã chấm công rồi thì không bị chấm trùng lặp
  
  
def reset_last_detection_data():
    global user_count
    detected_label.config(text='')
    user_count = {}
      

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
    global root
    root.quit()     
    root.destroy()

# Function to close the camera stream and window
def close_camera():
    global camera_opened, video_capture
    camera_opened = False
    if video_capture:
     video_capture.release()
    _quit()

# Function to confirm exit with "Ctrl + q"
def confirm_exit(event):
    print('Confirm exit')
    if event.keysym == 'q' and event.state == 4:  # Checking for 'q' and Ctrl key combination
        confirm = messagebox.askokcancel("Confirm Exit", "Are you sure you want to exit?")
        if confirm:
            close_camera()

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
# hello_label = Label(left_frame, text='Login', font=('Arial', 20))
# hello_label.grid(row=0, column=0, columnspan=2, pady=10)

# Login form components
# username_label = Label(left_frame, text='Username:')
# username_label.grid(row=1, column=0)
# username_entry = Entry(left_frame)
# username_entry.grid(row=1, column=1)

# password_label = Label(left_frame, text='Password:')
# password_label.grid(row=2, column=0)
# password_entry = Entry(left_frame, show='*')
# password_entry.grid(row=2, column=1)

# login_button = Button(left_frame, text='Login', command=login_clicked)
# login_button.grid(row=3, columnspan=2, pady=10)

# # Label in the right frame for displaying camera stream
label = Label(right_frame)
label.pack(fill='both', expand=True)

# smooth_camera_label = Label(right_frame)
# smooth_camera_label.pack(fill='both', expand=True)

log_label = Label(left_frame, text='Loading known_faces...')
log_label.grid(row=4, columnspan=2,  pady=10 )

detected_label_font = ('Arial', 14, 'bold')
detected_label = Label(left_frame, text='', font=detected_label_font) # fg='#FFA500'
detected_label.grid(row=5, columnspan=2,  pady=10 )

login_button = Button(left_frame, text='Reset detected data', command=reset_last_detection_data)
login_button.grid(row=6, columnspan=2, pady=10)

# # Start camera thread
camera_thread = threading.Thread(target=camera_thread)
camera_thread.start()

# # smooth_camera_thread
smooth_camera_thread = threading.Thread(target=smooth_camera)
smooth_camera_thread.start()

# compare_faces_thread = threading.Thread(target=compare_faces)
# compare_faces_thread.start()

# Bind closing event to the window
root.protocol("WM_DELETE_WINDOW", close_camera)

# Bind Ctrl + q to confirm exit
root.bind_all("<Control-q>", confirm_exit)

root.mainloop()


