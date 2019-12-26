import os
import numpy as np
import skimage.io
from keras.utils import np_utils

NUM_CLASSES = 7

def append_to_list(path,data_x,data_y,emition):
    files = os.listdir(path)
    files.sort()
    for filename in files:
        I = skimage.io.imread(os.path.join(path, filename))
        data_x.append(I.tolist())
        data_y.append(emition)

    return data_x,data_y

def duplicate_input_layer(array_input,size):
    model_input = np.empty([size,48,48,3])
    for index,iterm in enumerate(model_input):
        iterm[:,:,0] = array_input[index]
        iterm[:,:,1] = array_input[index]
        iterm[:,:,2] = array_input[index]
    return model_input

def get_ck_data(ck_path):
    anger_path = os.path.join(ck_path, 'anger')
    disgust_path = os.path.join(ck_path, 'disgust')
    fear_path = os.path.join(ck_path, 'fear')
    happy_path = os.path.join(ck_path, 'happy')
    sadness_path = os.path.join(ck_path, 'sadness')
    surprise_path = os.path.join(ck_path, 'surprise')
    contempt_path = os.path.join(ck_path, 'contempt')

    data_x = []
    data_y = []
    data_x, data_y = append_to_list(anger_path,data_x,data_y,0)
    data_x, data_y = append_to_list(disgust_path, data_x, data_y, 1)
    data_x, data_y = append_to_list(fear_path, data_x, data_y, 2)
    data_x, data_y = append_to_list(happy_path, data_x, data_y, 3)
    data_x, data_y = append_to_list(sadness_path, data_x, data_y, 4)
    data_x, data_y = append_to_list(surprise_path, data_x, data_y, 5)
    data_x, data_y = append_to_list(contempt_path, data_x, data_y, 6)

    np_image_array = np.array(data_x)
    np_image_array = np_image_array.astype('float32') / 255.0
    batch_pixels_all = duplicate_input_layer(np_image_array,int(len(np_image_array)))
    print(batch_pixels_all.shape, type(batch_pixels_all))

    y_data_categorical = np_utils.to_categorical(data_y, NUM_CLASSES)
    print(y_data_categorical.shape, type(y_data_categorical))

    return batch_pixels_all,y_data_categorical

