# ResNet model
import keras
import numpy as np
import time

from tensorflow.python.client import device_lib
#from tomo_challenge.utils.utils import save_logs
#from tomo_challenge.utils.utils import calculate_metrics
#from tomo_challenge.utils.utils import get_available_gpus

class Classifier_RESNET:

	def __init__(self, output_directory, input_shape, nb_classes, verbose=True,build=True,load_weights=False):
		self.output_directory = output_directory
		if build==True:
			self.model = self.build_model(input_shape, nb_classes)
			if(verbose==True):
				self.model.summary()
			self.verbose = verbose
			if load_weights == True:
				self.model.load_weights(self.output_directory
										.replace('resnet_augment','resnet')
										.replace('TSC_itr_augment_x_10','TSC_itr_10')
										+'/model_init.hdf5')
			else:
				self.model.save_weights(self.output_directory+'model_init.hdf5')
		return

	def build_model(self, input_shape, nb_classes):
		n_feature_maps = 64

		input_layer = keras.layers.Input(input_shape)

		# BLOCK 1

		conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
		conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
		conv_x = keras.layers.Activation('relu')(conv_x)

		conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
		conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
		conv_y = keras.layers.Activation('relu')(conv_y)

		conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
		conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

		# expand channels for the sum
		shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
		shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

		output_block_1 = keras.layers.add([shortcut_y, conv_z])
		output_block_1 = keras.layers.Activation('relu')(output_block_1)

		# BLOCK 2

		conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1)
		conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
		conv_x = keras.layers.Activation('relu')(conv_x)

		conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
		conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
		conv_y = keras.layers.Activation('relu')(conv_y)

		conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
		conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

		# expand channels for the sum
		shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1, padding='same')(output_block_1)
		shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

		output_block_2 = keras.layers.add([shortcut_y, conv_z])
		output_block_2 = keras.layers.Activation('relu')(output_block_2)

		# BLOCK 3

		conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_2)
		conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
		conv_x = keras.layers.Activation('relu')(conv_x)

		conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
		conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
		conv_y = keras.layers.Activation('relu')(conv_y)

		conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
		conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

		# no need to expand channels because they are equal
		shortcut_y = keras.layers.normalization.BatchNormalization()(output_block_2)

		output_block_3 = keras.layers.add([shortcut_y, conv_z])
		output_block_3 = keras.layers.Activation('relu')(output_block_3)

		# FINAL

		gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

		output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

		model = keras.models.Model(inputs=input_layer, outputs=output_layer)

		model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
			metrics=['accuracy'])

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

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
			verbose=self.verbose, callbacks=self.callbacks)

		duration = time.time() - start_time
		print(f'Duration - ResNet - Training = {duration:.2f}s')
		self.model.save(self.output_directory + 'last_model.hdf5')


		keras.backend.clear_session()


	def predict(self, x_test):
		model_path = self.output_directory.replace('ensemble_result', 'resnet')
		model_path = model_path + 'best_model.hdf5'
		model = keras.models.load_model(model_path)
		y_pred = model.predict(x_test)
		return  y_pred

def get_available_gpus():
	local_device_protos = device_lib.list_local_devices()
	return [x.name for x in local_device_protos if x.device_type == 'GPU']
