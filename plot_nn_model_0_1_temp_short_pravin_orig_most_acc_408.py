
#model = load_model('D:\\good_aud_model_char360_152_1_to_45_20211205-153214.h5') 

import numpy
from imblearn.over_sampling import SMOTE
from numpy import savetxt
from numpy import loadtxt
from matplotlib import pyplot
from pandas import DataFrame
from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from numpy import savetxt
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.losses import sparse_categorical_crossentropy
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
from keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import LSTM
from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from keras.utils import to_categorical 
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder
import dill as pickle
from sklearn.pipeline import Pipeline
from numpy import asarray
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import learning_curve
from sklearn.ensemble import StackingClassifier
from keras.layers import Input
from keras.models import Model
from keras.losses import binary_crossentropy
from tensorflow.keras import regularizers
from numpy.random import seed
from tensorflow.random import set_seed
import tensorflow
import time
from tensorflow.keras import layers
from tensorflow.keras.models import Model

# setting the seed
seed(1)
set_seed(1)

def NormalizeData(data):
	print(numpy.amin(data))	
	print(numpy.amax(data))	
	return (data + abs(numpy.amin(data))) / (numpy.amax(data) - numpy.amin(data))

# load array
X_train_whole = loadtxt('d:\\char360_408_1_to_45_py_128.csv', delimiter=',')


# augment data
choice = X_train_whole[:, -1] == 0.
X_total = numpy.append(X_train_whole, X_train_whole[choice, :], axis=0)
print(X_total.shape)

# balancing
sm = SMOTE(random_state = 2)
X_train_res, Y_train_res = sm.fit_resample(X_total, X_total[:, -1].ravel())
print("After OverSampling, counts of label '1': {}".format(sum(Y_train_res == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(Y_train_res == 0)))

tensorflow.compat.v1.reset_default_graph()
X_train, X_test, Y_train, Y_test = train_test_split(X_train_res, Y_train_res, random_state=1, test_size=0.3, shuffle = True)
print(X_train.shape)
print(X_test.shape)


#=======================================
 
# Model configuration

input = numpy.zeros((0, 408))
testinput = numpy.zeros((0, 408))

mean_of_train = mean(X_train[:, 0:408])
print(mean_of_train)
input = X_train[:, 0:408] - mean_of_train
too_high_input = input > 40.
input[too_high_input] = 40.
too_low_input = input < -40.
input[too_low_input] = -40.
input = NormalizeData(input)
input_output = numpy.append(input, Y_train.reshape(len(Y_train), 1), axis=1) 
savetxt('d:\\input_output.csv', input_output, delimiter=',')

mean_of_test = mean(X_test[:, 0:408])
print(mean_of_test)
testinput = X_test[:, 0:408] - mean_of_test
too_high_testinput = testinput > 40.
testinput[too_high_testinput] = 40.
too_low_testinput = testinput < -40.
testinput[too_low_testinput] = -40.
testinput = NormalizeData(testinput)
savetxt('d:\\testinput.csv', testinput, delimiter=',')
#=====================================

print(len(input))
print(len(testinput))

input = input.reshape(len(input), 4, 102)
input = input.transpose(0, 2, 1)
print (input.shape)

testinput = testinput.reshape(len(testinput), 4, 102)
testinput = testinput.transpose(0, 2, 1)
print (testinput.shape)


# Create the model
inputs = layers.Input(shape=(102, 4), name="inputs")
x = layers.Conv1D(filters=100, kernel_size=8, strides=3, activation='relu', name="conv1d_1") (inputs)
x = layers.MaxPooling1D(pool_size=2) (x)
x = layers.Conv1D(50, 4, activation='relu', name="conv1d_4") (x)
x = layers.Dropout(0.3, name="dropout_1") (x,  training=True)
x = layers.GlobalMaxPooling1D(name="gmp1d") (x)
x = layers.Dense(32, activation='relu', name="dense_1") (x)
x = layers.Dense(2, activation='softmax', name="dense_2") (x)
out = x    
model = Model(inputs, out)

model.summary()

# Compile the model
sgd = SGD(lr=0.007, momentum=0.7, nesterov=True)       # Higher values get steep loss curves ..
model.compile(loss=sparse_categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

hist = model.fit(input, Y_train, batch_size=32, epochs=100, verbose=1, validation_data=(testinput, Y_test), steps_per_epoch=None)	


# evaluate the model
Y_hat_temp = model.predict(testinput)
Y_hat_classes = numpy.argmax(Y_hat_temp, axis=1)
matrix = confusion_matrix(Y_test, Y_hat_classes)
print(matrix)


# plot training history
pyplot.plot(hist.history['loss'], label='tr_loss')
pyplot.plot(hist.history['val_loss'], label='val_loss')
pyplot.plot(hist.history['accuracy'], label='tr_accuracy')
pyplot.plot(hist.history['val_accuracy'], label='val_accuracy')
pyplot.legend()

pyplot.show()

#==================================

fileName = time.strftime("%Y%m%d-%H%M%S")	
model.save("D:\\most_acc_model_char360_152_1_to_45_" + fileName + ".h5")

#==================================

#Removed dropout and reduced momentum and reduced learning rate