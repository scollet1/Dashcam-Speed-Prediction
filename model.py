from keras import backend as K
K.set_image_dim_ordering('tf')

from keras.layers import LSTM
from keras.optimizers import Adam
from keras.models import Sequential, load_model
# from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.convolutional import Convolution2D as Conv2D
from keras.layers.core import Activation, Dropout, Flatten, Dense, Lambda

class Model:
	def __init__(self):
		self._model = None

	def summary(self):
		return self._model.summary()

	def create_model(self, in_shape, out_shape):
		model = Sequential()
		# model.add(Dense(input_shape = in_shape))
		# model.add(Lambda(lambda x: x / 127.5 - 1, input_shape = in_shape))
		model.add(ConvLSTM2D(24, (5, 5), input_shape=in_shape, return_sequences=True))
		model.add(BatchNormalization())
		model.add(ConvLSTM2D(24, (3, 3), return_sequences=True))
		model.add(Flatten())
		model.add(Dense(out_shape))
		optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
		self._model = model

	def predict_generator(self, gen, steps=1):
		return self._model.predict_generator(gen, steps=steps)

	def predict(self, X):
		return self._model.predict(X)

	def load(self, path):
		self._model = load_model(path)

	def save(self, path):
		self._model.save(path)

	def train(self, train_set, valid_set, valid_steps, steps=40, epochs=25, verbosity=1):
		# training history
		return self._model.fit_generator(
			train_set,
			steps_per_epoch=steps, 
			epochs=epochs,
			# callbacks=callbacks_list,
			verbose=verbosity,
			validation_data=valid_set,
			validation_steps=valid_steps
		)
