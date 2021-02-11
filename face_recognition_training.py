import cv2 as cv
import os
import numpy as np

path = 'C:\\Users\\igur\\PycharmProjects\\opencv_training\\face_detection\\faces'
image_size = (200, 200)


def read_images(path, image_size):
    name = []
    training_images, training_labels = [], []
    label = 0
    for dirname, subdirnames, filenames in os.walk(path):
        for subdirname in subdirnames:
            name.append(subdirname)
            images_path = os.path.join(dirname, subdirname)
            print('done.')
            print('dirname', dirname)

            for filename in os.listdir(images_path):
                img = cv.imread(os.path.join(images_path, filename), cv.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv.resize(img, image_size)
                training_images.append(img)
                training_labels.append(label)
            label += 1

    training_images = np.asarray(training_images, np.uint8)
    training_labels = np.asarray(training_labels, np.int32)

    return training_images, training_labels, name


train_data, label, name = read_images(path, image_size)

print(train_data.shape)
print('names: ', name)
print('labels:', label.shape)