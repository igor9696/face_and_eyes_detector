import numpy as np
import cv2 as cv
from face_detection import face_recognition_training as fr

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
path_to_images = 'C:\\Users\\igur\\PycharmProjects\\opencv_training\\face_detection\\faces\\'

# model training
train_data, target, name = fr.read_images(path_to_images, (200, 200))
model = cv.face.EigenFaceRecognizer_create()
model.train(train_data, target)

cap = cv.VideoCapture(0)
while True:
    _, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    for face in faces:
        x, y, w, h = face
        roi_gray = cv.resize(gray[y:y+h, x:x+w], (200, 200))
        label, confidence = model.predict(roi_gray)
        text = '%s, conf=%d' % (name[label], confidence)
        cv.putText(frame, text, (x, y-20), cv.FONT_HERSHEY_COMPLEX,0.8,[255,255,0],3)
        cv.rectangle(frame, (x, y), (x+w, y+h), [0,0,255], 4)

    cv.imshow('result', frame)
    if cv.waitKey(5) & 0xff == 27:
        break

cap.release()
cv.destroyAllWindows()

