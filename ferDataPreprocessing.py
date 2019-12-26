import pandas as pd
import numpy as np
from keras import backend as K
from keras.utils import np_utils
import random as rd

NUM_CLASSES = 7
IMG_SIZE = 48

def train_data_shuffle_batch(imags_all,labels_all,batch_size):
    index = []
    for i in range(batch_size):
        num = rd.randint(0,len(labels_all)-1)
        index.append(num)

    return imags_all[index],labels_all[index]

def process_emotion(emotion):
    emotion_as_list = [iterm[0] for iterm in emotion.values.tolist()]
    y_data = []
    for index in range(len(emotion_as_list)):
        y_data.append(emotion_as_list[index])

    y_data_categorical = np_utils.to_categorical(y_data, NUM_CLASSES)
    return y_data_categorical


def process_pixels(pixels, image_size=IMG_SIZE):
    pixels_as_list = [iterm[0] for iterm in pixels.values.tolist()]
    np_image_array = []
    for index, iterm in enumerate(pixels_as_list):
        data = np.zeros((image_size, image_size), dtype=np.uint8)
        pixel_data = iterm.split()

        for i in range(0, image_size):
            pixel_index = i * image_size
            data[i] = pixel_data[pixel_index:pixel_index + image_size]

        np_image_array.append(np.array(data))

    np_image_array = np.array(np_image_array)
    np_image_array = np_image_array.astype('float32') / 255.0
    return np_image_array

def duplicate_input_layer(array_input,size):
    model_input = np.empty([size,48,48,3])
    for index,iterm in enumerate(model_input):
        iterm[:,:,0] = array_input[index]
        iterm[:,:,1] = array_input[index]
        iterm[:,:,2] = array_input[index]
    return model_input

def raw_data_transform_batch(csv_file):
    raw_data = pd.read_csv(csv_file)
    emotion_array = process_emotion(raw_data[['emotion']])
    print(emotion_array.shape)

    pixels_array = process_pixels(raw_data[['pixels']])
    print(pixels_array.shape, type(pixels_array))

    batch_pixels_all = duplicate_input_layer(pixels_array, int(len(pixels_array)))
    print(batch_pixels_all.shape, type(batch_pixels_all))
    return batch_pixels_all,emotion_array