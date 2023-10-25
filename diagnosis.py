import numpy as np
import pickle
import streamlit as st

#loading The same model
Loded_model= pickle.load(open('E:/Diabetes Prediction/trained_model.sav', 'rb'))

#creating a function for prediction

def deibetis_predicition(input_data):


    data_np_array= np.asarray(input_data)
    data_shape= data_np_array.reshape(1,-1)
    pred= Loded_model.predict(data_shape)

    if(pred[0]==0):
        return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
  
def main():
    
    #giving a title
    st.title('Diabetes Prediction')
    
    #getting the input data from user
    
    Pregnancies= st.text_input("Number of Pregnancies")
    Glucose= st.text_input("Glucose Level")
    BloodPressure= st.text_input("BloodPressure Value")
    SkinThickness= st.text_input("Skin Thickness Value")
    Insulin= st.text_input("Insulin Level")
    BMI= st.text_input("BMI Value")
    DiabetesPedigreeFunction= st.text_input("Diabetes Pedigree Function Value")
    Age= st.text_input("Age of the Person")
    
    #code for Prediction
    diagnosis= ''
    
    #creating  a button for diagnosis
    if st.button('Diabetes test result'):
        diagnosis= deibetis_predicition([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    
    st.success(diagnosis)
    

if __name__== '__main__':
    main()    