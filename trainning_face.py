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
        # Display a countdown before capturing an image
        # countdown = 5
        # cv2.rectangle(frame, (10, 20), (200, 40), (0, 0, 0), -1)
        # while countdown > 0:
        #     # Clear the area where the countdown text will be displayed
        #     cv2.rectangle(frame, (10, 20), (200, 40), (0, 0, 0), -1)
        #     countdown_text = f"Capture in: {countdown}"
        #     cv2.putText(frame, countdown_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #     cv2.imshow('Face Detection', frame)
        #     time.sleep(1)
        #     countdown -= 1
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

        # Perform face detection on the grayscale frame
        # faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=14, minSize=(30, 30))

        face_locations = face_recognition.face_locations(gray_frame, number_of_times_to_upsample = 1, model='hog')

        for (top, right, bottom, left) in face_locations:
            # Draw rectangles around detected faces
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 165, 0), 2)

            if(last_second < 0 or last_second >= 59):
                last_second = cur_second
            # print(top, right, bottom, left)
            # print(last_second, cur_second)
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

        # # Capture a new frame for the next iteration
        # ret, frame = camera.read()
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    camera.release()
    cv2.destroyAllWindows()


# Function to upload and convert an image to grayscale
def upload_and_convert_image():
    filename = filedialog.askopenfilename(title='Select Image', filetypes=(('PNGs', '*.png'), ('JPGs', '*.jpg'), ('GIFs', '*.gif')))
    name = name_entry.get()
    if filename:
        # Load the image
        image = cv2.imread(filename)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Save the grayscale image to the training images folder
        grayscale_image_path = os.path.join(training_faces_dir, f'{name}_0.jpg')
        cv2.imwrite(grayscale_image_path, gray_image)

        # Display a message indicating successful conversion
        message_label = tk.Label(root, text='Image uploaded!')
        message_label.pack()



def update_button_states(arg):
    if name_entry.get():
        upload_button.config(state='normal')
        start_button.config(state='normal')
    else:
        upload_button.config(state='disabled')
        start_button.config(state='disabled')

# Create a window
root = tk.Tk()
root.title("Face Training")

# Get screen width and height for centering the window
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set the window size and position it at the center of the screen
window_width = int(screen_width * 0.3)
window_height = int(screen_height * 0.3)
window_x = (screen_width - window_width) // 2
window_y = (screen_height - window_height) // 2

root.geometry(f"{window_width}x{window_height}+{window_x}+{window_y}")  # Set window size and position



# Create a label and entry for name input
name_label = tk.Label(root, text="Enter Name:")
name_label.pack()

name_entry = tk.Entry(root)
name_entry.bind('<KeyRelease>', update_button_states)
name_entry.pack()

# Create a button to start face detection
start_button = tk.Button(root, text="Start Face Training", command=start_face_detection)
start_button.config(state='disabled')
start_button.pack()

# Create a button to upload and convert an image
upload_button = tk.Button(root, text='Upload Image', command=upload_and_convert_image)
upload_button.config(state='disabled')

upload_button.pack()

root.mainloop()


async def main():
    # Start the asynchronous tasks
    task_timer_func = asyncio.create_task(timer_func())
    task_display_camera = asyncio.create_task(display_camera_with_counter())

    # Wait for both tasks to complete
    await asyncio.gather(task_counter, task_display_camera)