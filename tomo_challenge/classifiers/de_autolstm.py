# AutoLSTM model for Deep Ensemble
import tensorflow.keras as keras
import numpy as np
import time 

from tomo_challenge.utils.utils import save_logs
from tomo_challenge.utils.utils import calculate_metrics
from tomo_challenge.utils.utils import get_available_gpus

class Classifier_LSTM:

	def __init__(self, output_directory, input_shape, nb_classes, verbose=True,build=True):
		self.output_directory = output_directory
		if build == True:
			self.model = self.build_model(input_shape, nb_classes)
			if(verbose==True):
				self.model.summary()
			self.verbose = verbose
			if 'autolstm' in self.output_directory:
				self.model.save_weights(self.output_directory+'model_init.hdf5')
		return

	def build_model(self, input_shape, nb_classes):
		input_layer = keras.layers.Input(input_shape)

		x = keras.layers.Conv1D(32, 5, padding='same', activation='relu')(input_layer)
		x = keras.layers.Conv1D(512, 5, padding='same', activation='relu')(x)
		x = keras.layers.Conv1D(128, 5, padding='same', activation='relu')(x)

		x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True), merge_mode='concat')(x)
		x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=False), merge_mode='concat')(x)

		x = keras.layers.Dense(512)(x)
		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.Activation('relu')(x)
		x = keras.layers.Dropout(0.25)(x)
		x = keras.layers.Dropout(0.5)(x)

		output_layer = keras.layers.Dense(nb_classes, activation='softmax')(x)

		model = keras.models.Model(inputs=input_layer, outputs=output_layer)

		model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(0.0005),
			metrics=['accuracy'])

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
			min_lr=0.0001)

		file_path = self.output_directory+'best_model.hdf5'

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)

		self.callbacks = [reduce_lr,model_checkpoint]

		return model 

	def fit(self, x_train, y_train):
		if len(get_available_gpus())==0:
			print('error')
			exit()
		# x_val and y_val are only used to monitor the test loss and NOT for training  
		batch_size = 512
		nb_epochs = 20

		mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

		start_time = time.time()
		self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
			verbose=1, callbacks=self.callbacks)
		
		duration = time.time() - start_time
		print(f'Duration - AutoLSTM - Training = {duration:.2f}s')
		self.model.save(self.output_directory+'last_model.hdf5')

		keras.backend.clear_session()

	def predict(self, x_test):
		model_path = self.output_directory.replace('ensemble_result', 'autolstm')
		model_path = model_path + 'best_model.hdf5'
		model = keras.models.load_model(model_path)
		y_pred = model.predict(x_test)

		return y_pred
