import numpy as np
import cv2 as cv
import os


face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# creating directory to load training images
output_folder = 'C:\\Users\\igur\\PycharmProjects\\opencv_training\\face_detection\\faces\\is\\'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def generate_face_data(frames_number=50):
    """Open video stream and take frames_number and save it to specified folder"""
    cap = cv.VideoCapture(0)
    count = 0
    while True:
        _, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(120, 120))

        for face in faces:
            x, y, w, h = face
            cv.rectangle(frame, (x, y), (x + w, y + h), [0, 0, 255], 4)
            face_images = cv.resize(gray[y:y + h, x:x + w], (200, 200))
            cv.imshow('faces', face_images)
            # save every single image into 'is' folder
            output_image = '%s%d.png' % (output_folder, count)
            cv.imwrite(output_image, face_images)
            count += 1

        cv.imshow('output', frame)

        if cv.waitKey(5) & 0xff == 27:
            break

        elif count >= frames_number:
            break

    cap.release()
    cv.destroyAllWindows()


generate_face_data()