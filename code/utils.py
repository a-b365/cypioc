import base64
import numpy as np
import io
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
from .colorizer import dlModel


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

def CompileModel(modelName):

	model = dlModel((224,224,1))
	X,Y = cvrt2lab()
	model.compile(optimizer="adam", loss="mse" , metrics=['accuracy'])
	history = model.fit(X,Y,epochs=100,batch_size=24)
	performanceGraph(history) #Creates performanceGraph
	model.save(modelName)

def loadModel(modelName):
	return tf.keras.models.load_model(modelName,compile=True)

def cvrt2rgb(filepath):
	#Test the Grayscale Image
	model = loadModel('../models/plaincnn.h5')
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
	img2_color = lab2rgb(result)
	img2 = tf.keras.utils.array_to_img(img2_color) #Returns a PIL Image Instance
	buffer = io.BytesIO() #Creates a memory buffer
	img2.save(buffer,format='JPEG') #Saves the image in buffer with supportable formats
	byte_im = buffer.getvalue()
	result = base64.b64encode(byte_im).decode('utf-8') #b64 encoding
	return result