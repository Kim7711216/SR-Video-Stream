import os
import requests
import cv2
import numpy as np
import numpy
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam

# Image Processing Functions
def modcrop(img, scale):
    tmpsz = img.shape
    sz = tmpsz[0:2]
    sz = sz - np.mod(sz, scale)
    img = img[0:sz[0], 1:sz[1]]
    return img

def shave(image, border):
    img = image[border: -border, border: -border]
    return img


# Defining SRCNN Model
def model():

    # define model type
    SRCNN = Sequential()

    # add model layers
    SRCNN.add(Conv2D(filters=128, kernel_size = (9, 9), kernel_initializer='glorot_uniform',
                     activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(filters=64, kernel_size = (3, 3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=1, kernel_size = (5, 5), kernel_initializer='glorot_uniform',
                     activation='linear', padding='valid', use_bias=True))

    # define optimizer
    adam = Adam(lr=0.0003)

    # compile model
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])

    return SRCNN

srcnn = model()
weights_path=r"E:\SR-Video-Stream\model\3051crop_weight_200.h5"
srcnn.load_weights(weights_path)

model_path="E:\SR-Video-Stream\model\model_1.h5"
model = tf.keras.models.load_model(model_path)

input_img_path=r"E:\SR-Video-Stream\lr_frame_imges\chunk_1\frame_1.jpg"
image = cv2.imread(input_img_path)
(h, w, c) = image.shape
# display the image width, height, and number of channels to our
print("width: {} pixels".format(w))
print("height: {}  pixels".format(h))
print("channels: {}".format(c))

cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

sr_img=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
sr_img = image / 255.0  # Normalize pixel values to [0, 1]
sr_img = np.expand_dims(sr_img, axis=0)
sr_imges = model.predict(sr_img)
sr_imges = np.clip(sr_imges, 0, 1)  # Clip pixel values to [0, 1]
sr_imges = np.squeeze(sr_imges)
sr_imges = (sr_imges * 255.0).astype(np.uint8)  # Scale pixel values back to [0, 255]
sr_imges = cv2.cvtColor(sr_imges, cv2.COLOR_RGB2BGR)
cv2.imshow('SR Image', sr_imges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# preprocess the image with modcrop
ref = modcrop(image, 3)

# convert the image to YCrCb - (srcnn trained on Y channel)
temp = cv2.cvtColor(ref, cv2.COLOR_BGR2YCrCb)

# create image slice and normalize
Y = numpy.zeros((1, temp.shape[0], temp.shape[1], 1), dtype=float)
Y[0, :, :, 0] = temp[:, :, 0].astype(float) / 255

# perform super-resolution with srcnn
pre = srcnn.predict(Y)

# post-process output
pre *= 255
pre[pre[:] > 255] = 255
pre[pre[:] < 0] = 0
pre = pre.astype(np.uint8)

# copy Y channel back to image and convert to BGR
temp = shave(temp, 6)
temp[:, :, 0] = pre[0, :, :, 0]
output = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)

cv2.imshow('Image', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
