from sklearn import preprocessing
from numpy import loadtxt
import numpy
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from numpy import savetxt
import keras.backend as K
from tensorflow.python.keras.backend import eager_learning_phase_scope
from numpy import mean

def NormalizeData(data):
    return (data + (40.0)) / (40.0 - (-40.0))

model = load_model('D:\\most_acc_model_char360_152_1_to_45_20211224-085223.h5') 


# for some model with dropout ...
f = K.function([model.inputs],
               [model.output])


def predict_with_uncertainty(f, x, no_classes, n_iter=50):
    result = numpy.zeros((n_iter,) + (x.shape[0], no_classes) )

    for i in range(n_iter):
        with eager_learning_phase_scope(value=1): # 0=test, 1=train
	        result[i] = numpy.array(f(x)).reshape(1440, 2)

    prediction = result.mean(axis=0)
    uncertainty = result.std(axis=0)
    return prediction, uncertainty    



#X = loadtxt('d:\\table_of_flashes_A_152_1_to_45.csv', delimiter=',')  # Good 
#X = loadtxt('d:\\table_of_flashes_E_152_1_to_45.csv', delimiter=',')  # Good 
#X = loadtxt('d:\\table_of_flashes_F_152_1_to_45.csv', delimiter=',')  # Good 


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

input = input.reshape(len(input), 4, 102)
input = input.transpose(0, 2, 1)


y_real = X[:, -1]

#y_pred, uncertainty = predict_with_uncertainty(f, input, 2, 50) 
y_pred = model.predict(input) 

#----------------------------------

y_corr = numpy.zeros((len(y_pred), 2))

for i in range(len(y_corr)):
	y_corr[i][1] = (y_pred[i][1] * 0.4)/((y_pred[i][1] * 0.4) + ((1 - y_pred[i][1]) * 1.6))
	y_corr[i][0] = ((1 - y_pred[i][1]) * 1.6)/((y_pred[i][1] * 0.4) + ((1 - y_pred[i][1]) * 1.6)) 

print(y_corr.shape)
print(y_corr)
#----------------------------------


y_max = numpy.argmax(y_pred, axis=1)
matrix = confusion_matrix(y_real, y_max)
print(matrix)











#y_max = numpy.zeros((len(y_corr), 1))

#for j in range(len(y_corr)):
#	y_max[j]=numpy.argmax(y_corr[j], axis=0)

#print(y_max.flatten().shape)
#print(y_max.flatten())
#print(y_real.shape)
#print(y_real)

#matrix = confusion_matrix(y_max.flatten().reshape(1440,), y_real.reshape(1440,))
#print(matrix)




