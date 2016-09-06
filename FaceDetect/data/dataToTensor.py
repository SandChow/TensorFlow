import os
from scipy.misc import imread
import pandas as pd

def dataToTensor(tensorFile):
	tensorFile.write("label, ")
	for i in range(4095):
		tensorFile.write("pixel" + str(i) + ", ")
	tensorFile.write("pixel4095\n")
	filePathFile = pd.read_csv('/sampa/home/sandchow/Desktop/TensorFlow/FaceDetect/data/trainFilePaths.csv')
	counter = 0
	for i in filePathFile.path:
		tensorFile.write(str(filePathFile.label[counter]) + ", ")
		counter += 1
		im = imread(i).flatten()
		for j in im[:-1]:
			tensorFile.write(str(j) + ", ")
		tensorFile.write(str(im[-1]) + '\n')


if __name__=="__main__":
	tensorFile = open("/sampa/home/sandchow/Desktop/TensorFlow/FaceDetect/data/train.csv", "wb")		
	dataToTensor(tensorFile)
	tensorFile.close()
