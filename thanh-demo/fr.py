# https://face-recognition.readthedocs.io/en/latest/face_recognition.html
# https://github.com/ageitgey/face_recognition
import cv2
import face_recognition

# Load known faces and their encodings
known_faces_encodings = []
known_faces_names = []

# Add known faces and their encodings
# Replace with your known faces and their encodings

# Access the webcam (change index if you have multiple cameras)
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, _frame0 = video_capture.read()
    # _frame = cv2.resize(_frame0, (160, 120))
    _frame = cv2.resize(_frame0, (320, 240))
    # _frame = cv2.resize(_frame0, (640, 480))
    frame = cv2.flip(_frame, 1)


    # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
    # rgb_frame = frame[:, :, ::-1]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample = 1)
    # faces = [(242, 407, 428, 221)]

    if face_locations:
        top = face_locations[0][0]  # Top coordinate
        right = face_locations[0][1]  # Right coordinate
        bottom = face_locations[0][2]  # Bottom coordinate
        left = face_locations[0][3]  # Left coordinate
        # detected_face = gray_frame[y:y+h, x:x+w]
        # detected_face = gray_frame[top:bottom, left:right]
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        print(face_locations)
        # print(face_encodings)

    

    # Loop through each face found in the frame
    for (top, right, bottom, left) in face_locations:
        # See if the face matches any known faces
        # matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)
        name = "Unknown"

        # Check if we found a match
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_faces_names[first_match_index]

        # Draw a box around the face and label with the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
