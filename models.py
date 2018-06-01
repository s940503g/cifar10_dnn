from keras.datasets import cifar10
from keras.layers import Input, Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.utils import to_categorical

def default_model():
	input = Input(shape=(32,32,3), name='custom_inputs')
	net = Conv2D(64, (3,3))(input)
	net = Conv2D(64, (1, 1))(net)
	net = MaxPooling2D((2,2))(net)
	net = Conv2D(128, (3,3))(net)
	net = Conv2D(128, (1, 1))(net)
	net = MaxPooling2D((2,2))(net)

	net = Flatten()(net)
	net = Dense(512)(net)
	net = Dense(512)(net)
	predictions = Dense(10, activation='softmax')(net)

	model = Model(inputs=input, outputs=predictions)
	model.summary()

	return model

def gabor_model():
	
	def gabor_weights(shape, dtype=None):
		import keras.backend as K
		import cv2
		import numpy as np
		kernels = None
		for i in range(shape[-1]):
			theta = np.pi*i / shape[-1]
			gf = cv2.getGaborKernel((shape[0], shape[1]), 1.0, theta, 2.0, 0.5, 0, ktype=cv2.CV_32F)
			gf = np.stack((gf,)*3, -1)
			gf = np.expand_dims(gf, -1)
			if np.any(kernels):
				kernels = np.concatenate((kernels, gf), -1)
			else:
				kernels = gf
		return kernels
		

	input = Input(shape=(32,32,3), name='custom_inputs')
	net = Conv2D(45, (5, 5), kernel_initializer=gabor_weights, use_bias=False, padding='same')(input)
	net = Conv2D(64, (3, 3), padding='same')(net)
	net = Conv2D(64, (1, 1), activation='relu', padding='same', use_bias=False)(net)
	net = MaxPooling2D((2, 2))(net)
	net = Conv2D(128, (3, 3), padding='same')(net)
	net = Conv2D(128, (1, 1), activation='relu', padding='same', use_bias=False)(net)
	net = Conv2D(128, (3, 3), padding='same')(net)
	net = Conv2D(128, (1, 1), activation='relu', padding='same', use_bias=False)(net)
	net = MaxPooling2D((2, 2))(net)

	net = Flatten()(net)
	net = Dense(512)(net)
	net = Dense(512)(net)
	predictions = Dense(10, activation='softmax')(net)

	model = Model(inputs=input, outputs=predictions)

	return model
if __name__ == "__main__":
	model = gabor_model()
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()

	(x_train, y_train), (x_test, y_test) = cifar10.load_data()

	y_train = to_categorical(y_train, 10)
	y_test = to_categorical(y_test, 10)

	from keras.preprocessing.image import ImageDataGenerator
	train_datagen = ImageDataGenerator(vertical_flip=True, horizontal_flip=True, rotation_range=180, rescale=1/255.)
	test_datagen = ImageDataGenerator(rescale=1/255.)

	train_data = train_datagen.flow(x_train, y_train, batch_size=128)
	val_data = test_datagen.flow(x_test, y_test, batch_size=128) 
	model.fit_generator(train_data, validation_data=val_data, verbose=1, epochs=50)
