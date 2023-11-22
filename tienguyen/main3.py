import cv2
import mediapipe as mp
import tkinter as tk
# from tkinter import filedialog
# from PIL import Image
import os
import face_recognition

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

# User data
user_data = {}

# GUI
root = tk.Tk()

# def add_user():
#     name = name_entry.get()
#     cap = cv2.VideoCapture(0)
#     while cap.isOpened():
#         success, image = cap.read()
#         if not success:
#             break

#         # Convert the BGR image to RGB
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = face_detection.process(image)

#         # Convert the image back to BGR color space
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         # Draw face detections
#         if results.detections:
#             for detection in results.detections:
#                 # Get the bounding box coordinates
#                 bbox = detection.location_data.relative_bounding_box
#                 xmin = int(bbox.xmin * image.shape[1])
#                 ymin = int(bbox.ymin * image.shape[0])
#                 width = int(bbox.width * image.shape[1])
#                 height = int(bbox.height * image.shape[0])

#                 # Get the score of the face detection
#                 score = detection.score[0]

#                 # Draw the bounding box and the score on the image
#                 cv2.rectangle(image, (xmin, ymin), (xmin + width, ymin + height), (0, 255, 0), 2)
#                 cv2.putText(image, f"{score:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Display the image
#         cv2.imshow('Capture User Face', image)
#         if cv2.waitKey(5) & 0xFF == 27:
#             break

#     cap.release()

# def add_user():
#     name = name_entry.get()
#     cap = cv2.VideoCapture(0)
#     while cap.isOpened():
#         success, image = cap.read()
#         if not success:
#             break

#         # Convert the BGR image to RGB
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = face_detection.process(image)

#         # Draw face detections
#         if results.detections:
#             for detection in results.detections:
#                 # Draw a box around the face
#                 x = int(detection.location_data.relative_bounding_box.xmin * image.shape[1])
#                 y = int(detection.location_data.relative_bounding_box.ymin * image.shape[0])
#                 w = int(detection.location_data.relative_bounding_box.width * image.shape[1])
#                 h = int(detection.location_data.relative_bounding_box.height * image.shape[0])
#                 cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

#                 cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 # Put the user's name below the box
#                 cv2.putText(image, name, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Display the image
#         cv2.imshow('Capture User Face', image)
#         if cv2.waitKey(5) & 0xFF == 27:
#             break

#         # Capture the user face when 'c' is pressed
#         if cv2.waitKey(5) & 0xFF == ord('c'):
#             # Save the image in the 'known_faces' folder
#             if not os.path.exists('known_faces'):
#                 os.makedirs('known_faces')
#             cv2.imwrite(f'known_faces/{name}.jpg', image)

#     cap.release()

def add_user():
    name = name_entry.get()
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Capture User Face', cv2.WINDOW_NORMAL)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # Convert the image back to BGR color space
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw face detections
        if results.detections:
            for detection in results.detections:
                # Draw a box around the face
                x = int(detection.location_data.relative_bounding_box.xmin * image.shape[1])
                y = int(detection.location_data.relative_bounding_box.ymin * image.shape[0])
                w = int(detection.location_data.relative_bounding_box.width * image.shape[1])
                h = int(detection.location_data.relative_bounding_box.height * image.shape[0])
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Put the user's name below the box
                cv2.putText(image, name, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the image
        cv2.resizeWindow('Capture User Face', 640, 360)
        cv2.imshow('Capture User Face', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

        # Capture the user face when 'c' is pressed
        if cv2.waitKey(5) & 0xFF == ord('c'):
            print('c is pressed')
            # Save the image in the 'known_faces' folder
            if not os.path.exists('known_faces'):
                os.makedirs('known_faces')
            cv2.imwrite(f'known_faces/{name}.jpg', image)
            cap.release()
            cv2.destroyWindow('Capture User Face')

# ... (rest of the code)

def identify_face(image):
    # Convert the image from BGR to RGB
    # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Find all face locations in the image
    face_locations = face_recognition.face_locations(image)

    # Convert the image back to BGR color space
    # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    for face_location in face_locations:
        print(face_location)
        top, right, bottom, left = face_location
        face_image = image[top:bottom, left:right]
        print(face_image)
        # Loop over all known faces
        for filename in os.listdir('known_faces'):
            if filename.endswith('.jpg'):
                user_name = os.path.splitext(filename)[0]
                print(user_name)
                user_image = cv2.imread(os.path.join('known_faces', filename))

                # Convert the user image from BGR to RGB
                rgb_user_image = cv2.cvtColor(user_image, cv2.COLOR_BGR2RGB)

                # Create a face encoding for the user image
                user_face_encoding = face_recognition.face_encodings(rgb_user_image)[0]

                # Find the face landmarks
                # face_landmarks = face_recognition.face_landmarks(face_image)

                # If face landmarks are found, compute the face encoding
                # if face_landmarks:
                face_encoding = face_recognition.face_encodings(image, known_face_locations=[face_location])[0]

                # Compare the face encoding with the user face encoding
                match = face_recognition.compare_faces([user_face_encoding], face_encoding)
                print(match)
                if match[0]:
                    return user_name

    return None

def detect_faces():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('MediaPipe Face Detection', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.resize(image, (640, 360))

        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # Convert the image back to BGR color space
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw face detections
        if results.detections:
            for detection in results.detections:
                # For now, we'll just use a placeholder function
                user_name = identify_face(image)

                # Convert the image back to BGR color space
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw a box around the face
                x = int(detection.location_data.relative_bounding_box.xmin * image.shape[1])
                y = int(detection.location_data.relative_bounding_box.ymin * image.shape[0])
                w = int(detection.location_data.relative_bounding_box.width * image.shape[1])
                h = int(detection.location_data.relative_bounding_box.height * image.shape[0])
                if user_name:
                    color = (0, 255, 0)  # Green
                else:
                    color = (0, 0, 255)  # Red
                    user_name = "unknown"
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                # Put the user's name below the box
                cv2.putText(image, user_name, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Display the image
        cv2.resizeWindow('MediaPipe Face Detection', 640, 360)
        cv2.imshow('MediaPipe Face Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()


name_label = tk.Label(root, text="Name")
name_label.pack()
name_entry = tk.Entry(root)
name_entry.pack()

add_button = tk.Button(root, text="Add User", command=add_user)
add_button.pack()

detect_button = tk.Button(root, text="Detect Faces", command=detect_faces)
detect_button.pack()

root.mainloop()
