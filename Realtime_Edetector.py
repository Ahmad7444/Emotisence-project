import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model(r'D:\CODING\EMOTISENCE\best_model.h5')

labels =  ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)


    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Crop the face region
        face_roi = gray_frame[y:y+h, x:x+w]
        
        # Resize to 48x48 (model input size)
        face_resized = cv2.resize(face_roi, (48, 48))
        
        # Normalize and reshape: (48,48,1) → (1,48,48,1)
        face_normalized = face_resized / 255.0
        face_input = face_normalized.reshape(1, 48, 48, 1)
        
        # Predict emotion
        predictions = model.predict(face_input)
        predicted_label = labels[np.argmax(predictions)]
        
        # Display predicted label
        cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
        

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()