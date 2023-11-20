import cv2
import numpy as np
import os

# Load pre-trained models for face detection and recognition
# detector = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
embedder = cv2.dnn.readNetFromTorch('openface.nn4.small2.v1.t7')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Directory containing images for face recognition
images_folder = 'known_faces'

# Function to extract face embeddings using OpenCV's DNN module
def get_face_embeddings(image):
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    # detector.setInput(blob)
    # detections = detector.forward()
    # Convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    # 
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=13, minSize=(100, 100))


    if len(faces) > 0:
        # Assuming only one face in the image for simplicity
        # face_box = detections[0, 0, 0, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
        # (startX, startY, endX, endY) = face_box.astype('int')

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = image[y:y+h, x:x+w]
            face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(face_blob)
            vec = embedder.forward()
            return vec.flatten()

        return None

    else:
        return None

# Load known face embeddings and names from the images folder
known_embeddings = []
known_names = []

for filename in os.listdir(images_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(images_folder, filename)
        img = cv2.imread(image_path)

        # Get embeddings for each known face
        face_embedding = get_face_embeddings(img)
        if face_embedding is not None:
            known_embeddings.append(face_embedding)
            # Extract name from file name (assuming naming convention)
            name = os.path.splitext(filename)[0]
            known_names.append(name)

# Function for face recognition using the known embeddings and names
def recognize_faces(frame):
    face_embedding = get_face_embeddings(frame)
    if face_embedding is not None:
        # Compare face embeddings with known faces
        distances = np.linalg.norm(np.array(known_embeddings) - face_embedding, axis=1)
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]

        # Set a threshold for face recognition
        if min_distance < 0.6:  # Adjust threshold as needed
            recognized_name = known_names[min_distance_idx]
            return recognized_name
        else:
            return 'Unknown'
    else:
        return 'Unknown'

# Access the camera feed
camera = cv2.VideoCapture(0)  # Use 0 for default webcam, change the index if using other cameras

while True:
    ret, _frame = camera.read()
    frame = cv2.flip(_frame, 1)


    if not ret:
        break

    recognized_name = recognize_faces(frame)

    # Display recognized name on the frame
    cv2.putText(frame, recognized_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the output frame
    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

camera.release()
cv2.destroyAllWindows()
