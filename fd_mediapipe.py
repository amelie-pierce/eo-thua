import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
from mediapipe.framework.formats import location_data_pb2
from media_pipe_utils import _normalized_to_pixel_coordinates
from typing import List, Mapping, Optional, Tuple, Union

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.9) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        # mp_drawing.draw_detection(image, detection)
        location = detection.location_data
        if location.HasField('relative_bounding_box'):
            image_rows, image_cols, _ = image.shape
            rbb_box = location.relative_bounding_box
            rect_start_point = _normalized_to_pixel_coordinates(
                rbb_box.xmin, rbb_box.ymin, image_cols,
                image_rows)
            rect_end_point = _normalized_to_pixel_coordinates(
                rbb_box.xmin + rbb_box.width,
                rbb_box.ymin + rbb_box.height, image_cols,
                image_rows)
            # // (263, 202) (475, 414)
            # print(rect_start_point, rect_end_point)
            cv2.rectangle(image, rect_start_point, rect_end_point,
                            (0, 0, 255), 2)
            
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()