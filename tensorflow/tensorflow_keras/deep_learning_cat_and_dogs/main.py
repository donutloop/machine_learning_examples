import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2
import random
import pickle
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

NAME = "Cats-vs-dog-cnn-64x1-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

def create_training_data(categories, datadir, img_size):
    
    training_data = []
    for category in categories:
        path = os.path.join(datadir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass 

    return training_data

CATEGORIES = ["Dog", "Cat"]
# https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765
DATADIR = "/home/user/workspace/python/machine_learning_examples/tensorflow/tensorflow_keras/deep_learning_cat_and_dogs/dataset/PetImages"

IMG_SIZE = 50

X = []
y = []

if os.path.exists("X.pickle") and os.path.exists("y.pickle"):

    pickle_x_in  = open("X.pickle", "rb")
    X = pickle.load(pickle_x_in)

    pickle_y_in = open("y.pickle", "rb")
    y = pickle.load(pickle_y_in)

if len(X) == 0 or len(y) == 0:

    training_data = create_training_data(CATEGORIES, DATADIR, IMG_SIZE)                
    random.shuffle(training_data)

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


X = X/255.0

model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X, y, batch_size=32, validation_split=0.1, callbacks=[tensorboard], epochs=10)

# run tensorboard --log logs to analyse logs

