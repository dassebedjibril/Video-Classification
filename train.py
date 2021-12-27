#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from imutils import paths
import re, os
import pickle


data_path = r"\home\phenomen\Workspace\TRANSPIR\Project_essay\data"
outputmodel = r"\home\phenomen\Workspace\TRANSPIR\Project_essay\video_classification_model\VideoClassificationModel"
outputlabelbinarizer = r"\home\phenomen\Workspace\TRANSPIR\Project_essay\video_classification_model\VideoClassificationModel"



Sports_labels = set(['boxing', 'swimming', 'table_tennis'])
print("images are being loaded.....")
path_to_images = list(paths.list_images(data_path))
data = []
labels = []

for images in path_to_images:
    label = images.split(os.path.sep)[-2]
    if label not in Sports_labels:
        continue
    image = cv2.imread(images)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (244, 244))
    data.append(image)
    labels.append(label)


data = np.array(data)
labels = np.array(labels)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
epoch = 25



from sklearn.model_selection import train_test_split




X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, stratify= labels, random_state=42)


# Data augmentation

from keras.preprocessing.image import ImageDataGenerator

TrainingAugmentation = ImageDataGenerator(
    rotation_range= 30, 
    zoom_range=0.15, 
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    shear_range=0.15,
    horizontal_flip= True,
    fill_mode= "nearest"
)

ValidationAugmentation = ImageDataGenerator()
mean = np.array([123.68, 116.779, 103.99], dtype = "float32")
TrainingAugmentation.meanan = mean
ValidationAugmentation.mean = mean



from keras.layers import Input
from keras.layers.pooling import AveragePooling2D
from tensorflow.keras.applications import ResNet50
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.models import Model
import tensorflow as tf


baseModel = ResNet50(weights= "imagenet", include_top = False, input_tensor= Input(shape = (224, 244, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size = (7,7))(headModel)
headModel = Flatten(name = "flatten")(headModel)
headModel = Dense(512, activation = "relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(lb.classes_), activation = "softmax")(headModel)

model = Model(inputs = baseModel.input, outputs = headModel)

for basemodellayers in baseModel.layers:
    basemodellayers.trainable = False


from tensorflow.keras.optimizers import SGD

opt = SGD(learning_rate = 0.0001, momentum= 0.9, decay = 1e-4/epoch)

model.compile(loss= "categorical_crossentropy", optimizer = opt, metrics= ["accuracy"])



History = model.fit_generator(
TrainingAugmentation.flow(X_train, y_train, batch_size= 32),
steps_per_epoch= len(X_train) // 32, 
validation_data= ValidationAugmentation.flow(X_test, y_test),
validation_steps = len(X_test) // 32,
epochs = epoch)


model.save(outputmodel)

lbbinarizer = open(r"\home\phenomen\Workspace\TRANSPIR\Project_essay\videoclassificationbinarizer.pickle", "wb") 
lbbinarizer.write(pickle.dumps(lb))
lbbinarizer.close()


