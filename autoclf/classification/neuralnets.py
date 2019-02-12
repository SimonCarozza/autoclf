from keras.models import Sequential
from keras.layers import Dense, Dropout

from random import randint

import warnings
warnings.filterwarnings("ignore")

### NNs for binary classification, no one-hot-encoding

def baseline_nn_model(input_dim):
	# create model
	bnn_model = Sequential()
	bnn_model.add(Dense(
		units=input_dim, input_dim=input_dim, activation='relu', 
		kernel_initializer='normal'))
	bnn_model.add(Dense(
		units=1, activation='sigmoid', kernel_initializer='normal'))
	# compile model
	bnn_model.compile(
		loss='binary_crossentropy', optimizer='adam', 
		metrics=['accuracy']
		)
	return bnn_model

def baseline_nn_smaller_model(input_dim):
	# create model
	bnn_model = Sequential()
	bnn_model.add(Dense(
		units=round(input_dim/2), input_dim=input_dim, activation='relu', 
		kernel_initializer='normal'))
	bnn_model.add(Dense(
		units=1, activation='sigmoid', kernel_initializer='normal'))
	# compile model
	bnn_model.compile(
		loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return bnn_model

def larger_nn_model(input_dim):
	# create model
	lnn_model = Sequential()
	lnn_model.add(Dense(
		units=input_dim*2, input_dim=input_dim, activation='relu', 
		kernel_initializer='normal'))
	lnn_model.add(Dense(
		units=input_dim, activation='relu', kernel_initializer='normal'))
	lnn_model.add(Dense(
		units=1, activation='sigmoid', kernel_initializer='normal'))
	# compile model
	lnn_model.compile(
		loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']
	 )
	# lnn_model.optimizer.lr = 0.01
	# lnn_model.summary()
	return lnn_model

def deep_nn_model(input_dim):
	# create model
	dnn_model = Sequential()
	dnn_model.add(Dense(
		units=input_dim*5, input_dim=input_dim, activation='relu', 
		kernel_initializer='normal'))
	dnn_model.add(Dense(
		units=input_dim*5, activation='relu', kernel_initializer='normal'))
	dnn_model.add(Dense(
		units=input_dim, activation='relu', kernel_initializer='normal'))
	dnn_model.add(Dense(
		units=1, activation='sigmoid', kernel_initializer='normal'))
	# compile model
	dnn_model.compile(loss='binary_crossentropy', optimizer='adam', 
		              metrics=['accuracy'])
	# dnn_model.summary()
	return dnn_model

def larger_deep_nn_model(input_dim):
	# create model
	dnn_model = Sequential()
	dnn_model.add(Dense(
		units=input_dim**3, input_dim=input_dim, activation='relu', 
		kernel_initializer='normal'))
	dnn_model.add(Dense(
		units=input_dim**2, activation='relu', kernel_initializer='normal'))
	dnn_model.add(Dense(
		units=input_dim, activation='relu', kernel_initializer='normal'))
	dnn_model.add(Dense(
		units=1, activation='sigmoid', kernel_initializer='normal'))
	# compile model
	dnn_model.compile(loss='binary_crossentropy', optimizer='adam', 
		              metrics=['accuracy'])
	# dnn_model.optimizer.lr = 0.01
	# dnn_model.summary()
	return dnn_model

def deeper_nn_model(input_dim):
	# create model
	dnn_model = Sequential()
	dnn_model.add(Dense(
		units=input_dim*5, input_dim=input_dim, activation='relu', 
		kernel_initializer='normal'))
	dnn_model.add(Dense(
		units=input_dim*3, activation='relu', kernel_initializer='normal'))
	dnn_model.add(Dense(
		units=input_dim*2, activation='relu', kernel_initializer='normal'))
	dnn_model.add(Dense(
		units=input_dim, activation='relu', kernel_initializer='normal'))
	dnn_model.add(Dense(
		units=1, activation='sigmoid', kernel_initializer='normal'))
	# compile model
	dnn_model.compile(loss='binary_crossentropy', optimizer='adam', 
		              metrics=['accuracy'])
	# dnn_model.summary()
	return dnn_model

# eventually, add units_3, drop_out_3
def tunable_deep_nn(
	input_dim, units_0, units_1, units_2,
	dropout_rate_0, dropout_rate_1, dropout_rate_2,
	):
	# create model
	nn_model = Sequential()
	nn_model.add(Dropout(dropout_rate_0))
	nn_model.add(Dense(
		units=units_0, input_dim=input_dim, activation='relu', 
		kernel_initializer='normal'))
	# hidden layers
	nn_model.add(Dropout(dropout_rate_1))
	nn_model.add(Dense(
		units=units_1, activation='relu', kernel_initializer='normal'))
	nn_model.add(Dropout(dropout_rate_2))
	nn_model.add(Dense(
		units=units_2, activation='relu', kernel_initializer='normal'))
	nn_model.add(Dense(
		units=1, activation='sigmoid', kernel_initializer='normal'))
	# compile model
	nn_model.compile(loss='binary_crossentropy', optimizer='adam', 
		              metrics=['accuracy'])
	# nn_model.optimizer.lr = 0.01
	# nn_model.summary()

	return nn_model

### multiclass
# metrics=['categorical_crossentropy']
# metrics=['categorical_accuracy']

def baseline_nn_model_multiclass(input_dim, output_dim):
	# create model
	bnn_model = Sequential()
	bnn_model.add(Dense(
		units=input_dim, input_dim=input_dim, activation='relu', 
		kernel_initializer='normal'))
	bnn_model.add(Dense(
		units=output_dim, activation='softmax', kernel_initializer='normal'))
	# compile model
	bnn_model.compile(loss='categorical_crossentropy', optimizer='adam', 
		              metrics=['accuracy'])
	# bnn_model.optimizer.lr = 0.01
	# bnn_model.summary()
	return bnn_model

def baseline_nn_smaller_model_multiclass(input_dim, output_dim):
	# create model
	bnn_model = Sequential()
	bnn_model.add(Dense(
		units=round(input_dim/2), input_dim=input_dim, activation='relu', 
		kernel_initializer='normal'))
	bnn_model.add(Dense(
		units=output_dim, activation='softmax', kernel_initializer='normal'))
	# compile model
	bnn_model.compile(
		loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# bnn_model.optimizer.lr = 0.01
	# bnn_model.summary()
	return bnn_model

def larger_nn_model_multiclass(input_dim, output_dim):
	# create model
	lnn_model = Sequential()
	lnn_model.add(Dense(
		units=input_dim*2, input_dim=input_dim, activation='relu', 
		kernel_initializer='normal'))
	lnn_model.add(Dense(
		units=input_dim, activation='relu', kernel_initializer='normal'))
	lnn_model.add(Dense(
		units=output_dim, activation='softmax', kernel_initializer='normal'))
	# compile model
	lnn_model.compile(
		loss='categorical_crossentropy', optimizer='adam', 
		metrics=['accuracy'])
	# lnn_model.optimizer.lr = 0.01
	# lnn_model.summary()
	return lnn_model

def deep_nn_model_multiclass(input_dim, output_dim):
	# create model
	dnn_model = Sequential()
	dnn_model.add(Dense(
		units=input_dim*5, input_dim=input_dim, activation='relu', 
		kernel_initializer='normal'))
	dnn_model.add(Dense(
		units=input_dim*5, activation='relu', kernel_initializer='normal'))
	dnn_model.add(
		Dense(units=input_dim, activation='relu', kernel_initializer='normal'))
	dnn_model.add(
		Dense(units=output_dim, activation='softmax', kernel_initializer='normal'))
	# compile model
	dnn_model.compile(loss='categorical_crossentropy', optimizer='adam', 
		metrics=['accuracy'])
	# lnn_model.optimizer.lr = 0.01
	# dnn_model.summary()
	return dnn_model

def larger_deep_nn_model_multiclass(input_dim, output_dim):
	# create model
	dnn_model = Sequential()
	dnn_model.add(Dense(
		units=input_dim**3, input_dim=input_dim, activation='relu', 
		kernel_initializer='normal'))
	dnn_model.add(Dense(
		units=input_dim**2, activation='relu', kernel_initializer='normal'))
	dnn_model.add(Dense(
		units=input_dim, activation='relu', kernel_initializer='normal'))
	dnn_model.add(Dense(
		units=output_dim, activation='softmax', kernel_initializer='normal'))
	# compile model
	dnn_model.compile(
		loss='categorical_crossentropy', optimizer='adam', 
		metrics=['accuracy'])
	# dnn_model.optimizer.lr = 0.01
	# dnn_model.summary()
	return dnn_model

def deeper_nn_model_multiclass(input_dim, output_dim):
	# create model
	dnn_model = Sequential()
	dnn_model.add(Dense(
		units=input_dim*5, input_dim=input_dim, activation='relu', 
		kernel_initializer='normal'))
	dnn_model.add(Dense(
		units=input_dim*3, activation='relu', kernel_initializer='normal'))
	dnn_model.add(
		Dense(units=input_dim*2, activation='relu', kernel_initializer='normal'))
	dnn_model.add(
		Dense(units=input_dim, activation='relu', kernel_initializer='normal'))
	dnn_model.add(
		Dense(units=output_dim, activation='softmax', kernel_initializer='normal'))
	# compile model
	dnn_model.compile(loss='categorical_crossentropy', optimizer='adam', 
		metrics=['accuracy'])
	# lnn_model.optimizer.lr = 0.01
	# dnn_model.summary()
	return dnn_model

def tunable_deep_nn_multiclass(
	input_dim, output_dim, units_0, units_1, units_2, 
	dropout_rate_0, dropout_rate_1, dropout_rate_2
	):
	# create model
	nn_model = Sequential()
	nn_model.add(Dropout(dropout_rate_0))
	nn_model.add(Dense(
		units=units_0, input_dim=input_dim, activation='relu', 
		kernel_initializer='normal'))
	# hidden layers
	nn_model.add(Dropout(dropout_rate_1))
	nn_model.add(Dense(
		units=units_1, activation='relu', kernel_initializer='normal'))
	nn_model.add(Dropout(dropout_rate_2))
	nn_model.add(Dense(
		units=units_2, activation='relu', kernel_initializer='normal'))
	nn_model.add(Dense(
		units=output_dim, activation='softmax', kernel_initializer='normal'))
	# compile model
	nn_model.compile(
		loss='categorical_crossentropy', optimizer='adam', 
		metrics=['accuracy'])
	# nn_model.optimizer.lr = 0.01
	# nn_model.summary()
	
	return nn_model