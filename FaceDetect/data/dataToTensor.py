import os
from scipy.misc import imread
import pandas as pd

dataToTensor(tensorFile):
	filePathFile = pd.read_csv('/sampa/home/sandchow/Desktop/TensorFlow/FaceDetect/data/trainFilePaths.csv')
	for i in filePathFile.path:
		im = imread(i).flatten()
		for j in im:
			tensorFile.write(j + ", ")
		tensorFile.write('\n')


if __name__=="__main__":
	tensorFile = open("/sampa/home/sandchow/Desktop/TensorFlow/FaceDetect/data/train.csv", "wb")		
	dataToTensor(tensorFile)
	tensorFile.close()