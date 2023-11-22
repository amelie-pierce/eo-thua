import mediapipe as mp
import cv2
import numpy as np

# Create a FaceDetector object
mp_face_detector = mp.solutions.face_detection
face_detector = mp_face_detector.FaceDetection(min_detection_confidence=0.5)

# Access the camera and capture frames
cap = cv2.VideoCapture(0)
while cap.isOpened():
  success, image = cap.read()
  if not success:
    break

  # Convert the image to RGB color space
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Pass the image to the FaceDetector object
  results = face_detector.process(image)

  # Convert the image back to BGR color space
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  # Get the face detection results
  if results.detections:
    # Get the first face detection result
    detection = results.detections[0]

    # Get the bounding box coordinates
    bbox = detection.location_data.relative_bounding_box
    xmin = int(bbox.xmin * image.shape[1])
    ymin = int(bbox.ymin * image.shape[0])
    width = int(bbox.width * image.shape[1])
    height = int(bbox.height * image.shape[0])

    # Get the score of the face detection
    score = detection.score[0]

    # Draw the bounding box and the score on the image
    cv2.rectangle(image, (xmin, ymin), (xmin + width, ymin + height), (0, 255, 0), 2)
    cv2.putText(image, f"{score:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

  # Display the image
  cv2.imshow('Face Detection', image)

  # Wait for a key press
  if cv2.waitKey(5) & 0xFF == 27:
    break

# Release the camera and close the windows
cap.release()
cv2.destroyAllWindows()
