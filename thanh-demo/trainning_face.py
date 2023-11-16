import cv2
import tkinter as tk
import os

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

    # Start the camera
    camera = cv2.VideoCapture(0)  # Use 0 for default webcam, change the index if using other cameras

    while count < total_images:
        # Capture frame from the camera
        ret, frame = camera.read()

        if not ret:
            break

        # Convert the frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform face detection
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Draw rectangles around detected faces
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Save the detected face as an image with the given name
            face_img = gray_frame[y:y+h, x:x+w]
            count += 1
            face_img_path = os.path.join(training_faces_dir, f'{name}_{count}.jpg')
            cv2.imwrite(face_img_path, face_img)

        # Display the progress label on the live camera feed
        progress_text = f'Progress: {count}/{total_images}'
        cv2.putText(frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with rectangles around detected faces
        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Pause for 1 second between each image capture
        cv2.waitKey(1000)

    camera.release()
    cv2.destroyAllWindows()

# Create a Tkinter window
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
name_entry.pack()

# Create a button to start face detection
start_button = tk.Button(root, text="Start Face Training", command=start_face_detection)
start_button.pack()

root.mainloop()
