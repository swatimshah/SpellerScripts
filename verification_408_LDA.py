from sklearn import preprocessing
from numpy import loadtxt
import numpy
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from numpy import savetxt
import keras.backend as K
from tensorflow.python.keras.backend import eager_learning_phase_scope
from numpy import mean
import pickle

def NormalizeData(data):
    return (data + (40.0)) / (40.0 - (-40.0))

model_file = open ("LDA_model_408.sav", "rb")
model = pickle.load(model_file)


#X = loadtxt('d:\\table_of_flashes_408_1_to_45_20211208-100526.csv', delimiter=',') #Good
#X = loadtxt('d:\\table_of_flashes_408_1_to_45_20211209-013305.csv', delimiter=',')  #Good

#X = loadtxt('d:\\table_of_flashes_A_408_1_to_45.csv', delimiter=',') #Good
#X = loadtxt('d:\\table_of_flashes_E_408_1_to_45.csv', delimiter=',')  #Good
#X = loadtxt('d:\\table_of_flashes_F_408_1_to_45.csv', delimiter=',')  #Good
X = loadtxt('d:\\table_of_flashes_G_408_1_to_45.csv', delimiter=',')  #Good

mean_of_test = mean(X[:, 0:408])
print(mean_of_test)
input = X[:, 0:408] - mean_of_test
too_high_input = input > 40.
input[too_high_input] = 40.
too_low_input = input < -40.
input[too_low_input] = -40.
input = NormalizeData(input)
savetxt('d:\\input-swati-online.csv', input, delimiter=',')


y_real = X[:, -1]

y_pred = model.predict(input) 


#----------------------------------

#y_corr = numpy.zeros((len(y_pred), 2))

#for i in range(len(y_corr)):
#	y_corr[i][1] = (y_pred[i][1] * 0.4)/((y_pred[i][1] * 0.4) + ((1 - y_pred[i][1]) * 1.6))
#	y_corr[i][0] = ((1 - y_pred[i][1]) * 1.6)/((y_pred[i][1] * 0.4) + ((1 - y_pred[i][1]) * 1.6)) 

#print(y_corr.shape)
#print(y_corr)
#----------------------------------

#y_max = numpy.argmax(y_corr, axis=1)
matrix = confusion_matrix(y_real, y_pred)
print(matrix)




