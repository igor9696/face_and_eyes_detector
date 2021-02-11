import numpy as np
import cv2 as cv
from face_detection import face_recognition_training as fr

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
path_to_images = 'C:\\Users\\igur\\PycharmProjects\\opencv_training\\face_detection\\faces\\'
train_data, target, name = fr.read_images(path_to_images, (200, 200))

# face recognition models
#model = cv.face.EigenFaceRecognizer_create()
#model = cv.face.FisherFaceRecognizer_create()

# confidence score in LBPH: good match (>80), bad match(<50)
# confidence score in EigenFace: good match (4000-5000), bad match (>5000)

model = cv.face.LBPHFaceRecognizer_create()
model.train(train_data, target)

cap = cv.VideoCapture(0)
while True:
    t1 = cv.getTickCount()
    _, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    for face in faces:
        x, y, w, h = face
        roi_gray = cv.resize(gray[y:y+h, x:x+w], (200, 200))
        label, confidence = model.predict(roi_gray)
        text = '%s, conf=%d' % (name[label], confidence)
        # if confidence is high enough print green rectangle, otherwise print 'Unknown' with red color
        if confidence <= 60:
            cv.rectangle(frame, (x, y), (x+w, y+h), [0,255,0], 4)
            cv.putText(frame, text, (x, y - 20), cv.FONT_HERSHEY_COMPLEX, 0.8, [0, 255, 0], 3)
        else:
            cv.rectangle(frame, (x, y), (x+w, y+h), [0,0,255], 4)
            cv.putText(frame, 'Unknown', (x, y - 20), cv.FONT_HERSHEY_COMPLEX, 0.8, [0, 0, 255], 3)

    # print FPS
    fps = int(cv.getTickFrequency()/(cv.getTickCount() - t1))
    cv.putText(frame, 'FPS: ' + str(fps),(50,50), cv.FONT_HERSHEY_COMPLEX, 0.7, [0, 0, 255], 3)

    # display result
    cv.imshow('result', frame)
    if cv.waitKey(5) & 0xff == 27:
        break

cap.release()
cv.destroyAllWindows()

