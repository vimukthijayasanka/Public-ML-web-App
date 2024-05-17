import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('diabetes_ml_model.sav','rb'))

def diabetes_prediction (input_data):

    input_data_array = np.asarray([input_data])
    # reshape error
    data_point = input_data_array.reshape(1, -1)

    prediction = loaded_model.predict(data_point)
    if (prediction[0] == 0):
        return("person is not a diabetes")
    else:
        return("person is a diabetes")

def main ():
    #giving a title for web page
    st.title("Diabetes Prediction Web App")

    #getting input data from the user
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose level")
    BloodPressure = st.text_input("Blood Pressure level")
    SkinThickness= st.text_input("Skin thickness level")
    Insulin = st.text_input("Insulin level")
    BMI = st.text_input("BMI value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Prediction Function value")
    Age = st.text_input("Age of the person")

    #code for prediction
    diagnosis=""

    #Creating a Button for prediction
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

    st.success(diagnosis)




