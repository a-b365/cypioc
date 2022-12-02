import tensorflow as tf
import matplotlib.pyplot as plt
import os.path
import base64
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
from sklearn.metrics import confusion_matrix
from skimage.io import imsave,imshow
from .colorizer import dlModel
from keras.models import Sequential
import cv2


def preprocess(path):
	augs_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1. / 255, validation_split=0.1)
	train = augs_datagen.flow_from_directory(path,target_size=(224,224),batch_size=24,class_mode=None,shuffle=True,subset='training')
	return train

def cvrt2lab():
	X =[]
	Y =[]
	for img in preprocess('../../landscapes/')[0]:
		try:
			lab = rgb2lab(img)
			X.append(lab[:,:,0])
			Y.append(lab[:,:,1:]/128)
		except:
			print('Error Occured while converting RGB to LAB color space.')

	X = np.array(X)
	Y = np.array(Y)
	X = X.reshape(X.shape+(1,))
	return X,Y

def performanceGraph(history):
    
    accuracy = history.history['accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    val_accuracy = history.history['val_accuracy']
    
    epochs_range=range(10)

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accuracy, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def CompileModel(optimizer,loss):

	model = dlModel((224,224,1))
	X,Y = cvrt2lab()
	model.compile(optimizer="adam", loss="mse" , metrics=['accuracy'])
	history = model.fit(X,Y,epochs=100,batch_size=24)
	performanceGraph(history) #Creates performanceGraph
	model.save('models/plaincnn.h5')


def cvrt2rgb(filepath):
	#Test the Grayscale Image
	model = tf.keras.models.load_model('../models/plaincnn.h5',compile=True)
	img1_color=[]
	img1 = tf.keras.utils.img_to_array(tf.keras.utils.load_img(filepath))
	img1 = resize(img1 ,(224,224))
	img1_color.append(img1)
	img1_color = np.array(img1_color, dtype= float)
	img1_color = rgb2lab(1.0/255*img1_color)[:,:,:,0]
	img1_color = img1_color.reshape(img1_color.shape+(1, ))
	output1 = model.predict(img1_color)
	output1 = output1*128
	result = np.zeros((224, 224, 3))
	result[:,:,0] = img1_color[0][:,:,0]
	result[:,:,1:] = output1[0]
	img1 = tf.keras.utils.array_to_img(result[:,:,1:])
	img1.show()
	#vis2 = cv2.cvtColor(result[:,:,1:], cv2.COLOR_GRAY2BGR)
	#_, buffer = cv2.imencode('.png', )
	#result = base64.b64encode(buffer).decode('utf-8')
	#return result