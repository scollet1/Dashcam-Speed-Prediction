import os
import re
import cv2
import sys
import random
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import Model
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
# from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator

#const
def RNN_STEPS():
	return 2

def PRE_PATH_TEST():
	return "data/test/images/"

def PRE_PATH_TRAIN():
	return "data/train/"
#end const

def calculate_optical_flow(a, b):
	grayscale_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
	grayscale_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
	hsv = np.zeros((66, 220, 3))
	hsv[:,:,1] = cv2.cvtColor(b, cv2.COLOR_RGB2HSV)[:,:,1]
	optical_flow = cv2.calcOpticalFlowFarneback(
		grayscale_a, grayscale_b,
		None, 0.5, 1, 15, 2, 5, 1.3, 0
	)
	mag, ang = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
	hsv[:,:,0] = ang * (180 / np.pi / 2)
	hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	hsv = np.asarray(hsv, dtype= np.float32)
	rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

	return rgb_flow

def preprocess_image(image):
	image_cropped = image[100:440, :-90]
	image = cv2.resize(image_cropped, (220, 66), interpolation = cv2.INTER_AREA)
	return image

def change_brightness(image, bright_factor):
	hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	hsv_image[:,:,2] = hsv_image[:,:,2] * bright_factor
	image_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
	return image_rgb

def preprocess_image_from_path(image_path, speed, bright_factor=None):
	# print("Loading image:", image_path)
	img = cv2.imread(image_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	if bright_factor:
		img = change_brightness(img, bright_factor)    
	img = preprocess_image(img)
	return img, speed

def load_frames(path):
	frame_files = os.listdir(path)
	frame_files.sort(key=lambda f: int(re.sub('\D', '', f)))
	return frame_files

def load_data(path):
	dframe = pd.read_csv('data/train.txt', sep="\n", header=0)
	dframe.insert(1, "frame", value=load_frames(path))
	return dframe

def split_valid_train_data(data, seed):
	train_data = data.iloc[:len(data) - (len(data) // 8)]
	valid_data = data.iloc[len(train_data):]
	return train_data, valid_data

def generate_train_set(data, batch_size=5):
	if batch_size < RNN_STEPS():
		batch_size = RNN_STEPS()
	image_batch = np.zeros((batch_size, RNN_STEPS(), 66, 220, 3))
	label_batch = np.zeros((batch_size))
	while True:
		for i in range(batch_size):
			idx = np.random.randint(1 + RNN_STEPS(), len(data) - batch_size)
			mean_speed = []
			for t in range(RNN_STEPS()):
				bright_factor = 0.2 + np.random.uniform()
	
				step = data.iloc[[idx - t]].reset_index()
				step_prev = data.iloc[[idx - t - 1]].reset_index()
				time_now = step['index'].values[0]
				time_prev = step_prev['index'].values[0]

				X, Y = preprocess_image_from_path(
					PRE_PATH_TRAIN() + step['frame'].values[0],
					step['speed'].values[0], bright_factor
				)
				x, y = preprocess_image_from_path(
					PRE_PATH_TRAIN() + step_prev['frame'].values[0],
					step_prev['speed'].values[0], bright_factor
				)
				rgb_diff = calculate_optical_flow(X, x)
				mean_speed.append(np.mean([Y, y]))

				image_batch[i][t] = rgb_diff
			label_batch[i] = np.mean(mean_speed)
		yield shuffle(image_batch, label_batch)

def generate_valid_set(data):
	while True:
		for idx in range(1 + RNN_STEPS(), len(data)):
			validate = np.zeros((1, RNN_STEPS(), 66, 220, 3))
			mean_speed = []
			for t in range(RNN_STEPS()):
				step = data.iloc[[idx - t]].reset_index()
				step_prev = data.iloc[[idx - t - 1]].reset_index()
				time_now = step['index'].values[0]
				time_prev = step_prev['index'].values[0]

				X, Y = preprocess_image_from_path(
					PRE_PATH_TRAIN() + step['frame'].values[0],
					step['speed'].values[0]
				)
				x, y = preprocess_image_from_path(
					PRE_PATH_TRAIN() + step_prev['frame'].values[0],
					step_prev['speed'].values[0]
				)
				img_diff = calculate_optical_flow(X, x)
				validate[0][t] = img_diff
				mean_speed.append(np.mean([Y, y]))
			yield validate, np.array([np.mean(mean_speed)])

def predict_testing_data(model):
	results = []
	data = pd.DataFrame({"frame": load_frames(PRE_PATH_TEST())})
	print(data)
	for i in range(len(data)):
		print("predicting frame:", data.iloc[[i]]['frame'].values[0])
		test_data = np.zeros((1, RNN_STEPS(), 66, 220, 3))
		if i > RNN_STEPS():
			for t in range(RNN_STEPS()):
				X, Y = preprocess_image_from_path(
					PRE_PATH_TEST() + data.iloc[[i - t]]['frame'].values[0], 0
				)
				x, y = preprocess_image_from_path(
					PRE_PATH_TEST() + data.iloc[[i - t - 1]]['frame'].values[0], 0
				)
				img_diff = calculate_optical_flow(X, x)
				test_data[0][t] = img_diff
		elif i > 0:
			X, Y = preprocess_image_from_path(
				PRE_PATH_TEST() + data.iloc[[i]]['frame'].values[0], 0
			)
			x, y = preprocess_image_from_path(
				PRE_PATH_TEST() + data.iloc[[i - 1]]['frame'].values[0], 0
			)
			img_diff = calculate_optical_flow(X, x)
			for t in range(RNN_STEPS()):
				test_data[0][t] = img_diff
		else:
			X, Y = preprocess_image_from_path(
				PRE_PATH_TEST() + data.iloc[[i]]['frame'].values[0], 0
			)
			for t in range(RNN_STEPS()):
				test_data[0][t] = X
		results.append(model.predict(test_data).tolist()[0])
	return list(itertools.chain.from_iterable(x for x in results))

if __name__=="__main__":
	if len(sys.argv) == 2:
		model = Model()
		if 'model.h5' in os.listdir('./'):
			model.load('model.h5')
		else:
			model.create_model(in_shape=(RNN_STEPS(), 66, 220, 3), out_shape=1)
		print(model.summary())
		if sys.argv[1] == 'predict':
			if not 'test.txt' in os.listdir('./'):
				results = predict_testing_data(model)
				with open('test.txt', 'w+') as f:
					f.write('\n'.join(map(str, results))) 
				plt.plot(results)
				plt.title('Speed Per Frame')
				plt.ylabel('Speed (MPH)')
				plt.xlabel('Frame')
				plt.show()
		elif sys.argv[1] == 'train':
			seed = random.seed(datetime.now())
			data = load_data(PRE_PATH_TRAIN())
			train, valid = split_valid_train_data(data, seed)
			print(train, valid)
			# plot_model(model, to_file='model.png')
			history = model.train(
				train_set=generate_train_set(train),
				valid_set=generate_valid_set(valid),
				valid_steps=len(valid.index),
				steps=30, # 25
				epochs=25, # 15
				verbosity=1
			)
			model.save('model.h5')
			
			print(history)
			
			plt.plot(history.history['mean_squared_error'])
			plt.title('Training History')
			plt.ylabel('Accuracy')
			plt.xlabel('Epoch')
			plt.legend(['MSE'], loc='upper right')
			plt.show()

