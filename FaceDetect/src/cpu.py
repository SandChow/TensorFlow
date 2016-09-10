import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv('../data/train.csv')
images = data.iloc[:,1:].values
images = images.astype(np.float)
# convert from [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)
image_size = images.shape[1]
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
