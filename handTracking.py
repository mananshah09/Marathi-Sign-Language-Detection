import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
import time

#for webcam as an input
cap = cv2.VideoCapture(0)
prevTime = 0
with mp_hands.Hands(
    min_detection_confidence=0.5,#detection sensitivity
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
      success, image = cap.read()
      if not success:
          print("Ignoring empty camera frame.")
          continue

      #flip the image horizontally for a selfie view display and convert BGR to RGB
      image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
      #to improve performance optionally mark the image as not writeable to pass by reference
      image.flags.writeable = False
      results = hands.process(image)

      #drawing hand annotations
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
      currTime = time.time()
      fps = 1 / (currTime - prevTime)
      prevTime = currTime
      #cv2.putText(image, f'FPS: {int(fps)}', (28,70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
      cv2.imshow('Hand Tracking', image)
      if cv2.waitKey(5) & 0xFF == 27:
          break
cap.release()