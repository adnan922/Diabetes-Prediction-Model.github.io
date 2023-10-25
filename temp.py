import numpy as np
import pickle

#loading The same model
Loded_model= pickle.load(open('E:/Diabetes Prediction/trained_model.sav', 'rb'))

data = (1, 89, 66, 23, 94, 28.1, 0.167, 21)

data_np_array= np.asarray(data)
data_shape= data_np_array.reshape(1,-1)
pred= Loded_model.predict(data_shape)

if(pred[0]==0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')