import numpy as np
import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

cap = cv.VideoCapture(1)

while True:
    _, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.blur(gray, (3,3))
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), [0, 0, 255], 5)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eyes_cascade.detectMultiScale(roi_gray)
        for xe, ye, we, he in eyes:
            cv.rectangle(roi_color, (xe, ye), (xe+we, ye+he), [255, 0, 0], 5)

    cv.imshow('res', frame)
    if cv.waitKey(5) & 0xff == 27:
        break

cap.release()
cv.destroyAllWindows()