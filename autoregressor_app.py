import streamlit as st

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline

import tensorflow
print (tensorflow.__version__)

st.header("Welcome to the Generative Playground")
st.write("This is an autoregressor model on cifar10 data set, with 50 epochs and 16 batch size trained only. RTX GPU is used to train the model.")

from tensorflow.keras.datasets import mnist,cifar10


(trainX, trainy), (testX, testy) = cifar10.load_data()

print('Training data shapes: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Testing data shapes: X=%s, y=%s' % (testX.shape, testy.shape))



for k in range(4):
    fig = plt.figure(figsize=(9,6))
    for j in range(9):
        i = np.random.randint(0, 10000)
        plt.subplot(990 + 1 + j)
        plt.imshow(trainX[i], cmap='gray_r')
        # st.pyplot(fig)
        plt.axis('off')
        #plt.title(trainy[i])
    plt.show()
    st.pyplot(fig)


# asdfaf

trainX = np.where(trainX < (0.33 * 256), 0, 1)
train_data = trainX.astype(np.float32)

testX = np.where(testX < (0.33 * 256), 0, 1)
test_data = testX.astype(np.float32)

train_data = np.reshape(train_data, (50000, 32, 32, 3))
test_data = np.reshape(test_data, (10000, 32, 32, 3))

print (train_data.shape, test_data.shape)


import tensorflow

class PixelConvLayer(tensorflow.keras.layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super(PixelConvLayer, self).__init__()
        self.mask_type = mask_type
        self.conv = tensorflow.keras.layers.Conv2D(**kwargs)

    def build(self, input_shape):
        # Build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)
        # Use the initialized kernel to create the mask
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)


# Next, we build our residual block layer.
# This is just a normal residual block, but based on the PixelConvLayer.
class ResidualBlock(tensorflow.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = tensorflow.keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )
        self.pixel_conv = PixelConvLayer(
            mask_type="B",
            filters=filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same",
        )
        self.conv2 = tensorflow.keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return tensorflow.keras.layers.add([inputs, x])

inputs = tensorflow.keras.Input(shape=(32,32,3))
x = PixelConvLayer(
    mask_type="A", filters=128, kernel_size=7, activation="relu", padding="same"
)(inputs)

for _ in range(5):
    x = ResidualBlock(filters=128)(x)

for _ in range(2):
    x = PixelConvLayer(
        mask_type="B",
        filters=128,
        kernel_size=1,
        strides=1,
        activation="relu",
        padding="valid",
    )(x)

out = tensorflow.keras.layers.Conv2D(
    filters=3, kernel_size=1, strides=1, activation="sigmoid", padding="valid"
)(x)

pixel_cnn = tensorflow.keras.Model(inputs, out)
pixel_cnn.summary()

adam = tensorflow.keras.optimizers.Adam(learning_rate=0.0005)
pixel_cnn.compile(optimizer=adam, loss="binary_crossentropy")


# %%
import os
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


pixel_cnn.load_weights(checkpoint_path)


# %% [markdown]
# # Display Results 81 images

# %%
#from IPython.display import Image, display
from tqdm import tqdm_notebook


# Create an empty array of pixels.
batch = 81
pixels = np.zeros(shape=(batch,) + (pixel_cnn.input_shape)[1:])
batch, rows, cols, channels = pixels.shape

print(pixels.shape)


import time 

# progress_text = "Operation in progress. Please wait."
# my_bar = st.progress(0, progress_text)
st.caption("Generating..... pls.. wait.. :)")
my_bar = st.progress(0)


# Iterate over the pixels because generation has to be done sequentially pixel by pixel.
for row in range(rows):
    for col in range(cols):
        for channel in range(channels):
            time.sleep(0.01)
            # Feed the whole array and retrieving the pixel value probabilities for the next
            # pixel.
            probs = pixel_cnn.predict(pixels)[:, row, col, channel]
            # Use the probabilities to pick pixel values and append the values to the image
            # frame.
            pixels[:, row, col, channel] = tensorflow.math.ceil(
              probs - tensorflow.random.uniform(probs.shape)
            )
    my_bar.progress(row+1)
time.sleep(1)

counter = 0
for i in range(4):
    figout = plt.figure(figsize=(9,6))
    for j in range(9):
        plt.subplot(990 + 1 + j)
        plt.imshow(pixels[counter,:,:,0])#, cmap='gray_r')
        counter += 1
        plt.axis('off')
    plt.show()
    st.pyplot(figout)

# %%



