import numpy as np
from tensorflow.keras.preprocessing import image

def preprocess_image(path):
    img = image.load_img(path, target_size=(32,32))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img