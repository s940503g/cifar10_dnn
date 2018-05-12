from keras.datasets import cifar10
from keras.layers import Input, Dense, GlobalAveragePooling2D, Activation
from keras.applications import vgg16
from keras.models import Model
from keras.utils import to_categorical
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = vgg16.preprocess_input(x_train.astype('float64'))
x_test = vgg16.preprocess_input(x_test.astype('float64'))

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

base_model = vgg16.VGG16(include_top=False, weights='imagenet')

input = Input(shape=(32,32,3), name='custom_inputs')
net = base_model(input)
net = GlobalAveragePooling2D()(net)
predictions = Dense(10, activation='softmax')(net)

model = Model(inputs=input, outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()
model.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=1)
