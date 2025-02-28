# importing the libaries


import numpy as np
import pickle # to load the model
import streamlit as st
import pandas as pd
import requests



# LOADING THE MODEL FROM  GITHUB
url = 'https://github.com/ezekielmose/model_sample/blob/main/strock_model_new.pkl'

# Download the file
loaded_model = requests.get(url)

# Save the downloaded content to a temporary file
with open('trained_model1.sav', 'wb') as f:
    pickle.dump(loaded_model.content, f)


# Load the saved model
with open('trained_model1.sav', 'rb') as f:
    loaded_model = pickle.load(f)


# Now, you can use the loaded model for predictions
# creating a function for prediction
def income_evaluation_prediction(input_data):

    input_data_as_numpy_array= np.array(input_data)
# reshaping the array for predicting 

    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # instead of 'model' we use loaded_model
    prediction = loaded_model.predict(input_data_reshaped)


    if prediction [0] == 0:
        return "The Person Earns less than 50K"
    else:
        return "The Person Earns more than 50K" # insted of print change to return  
    
# Streamlit library to craete a user interface
def main():
    
    # Interface title
    st.title("Income Evaluation Prediction Machine Learning Model")
    
    #getting the input data from the user  
    age = st.text_input("Enter the Person's Age 17 - 90")
    fnlwgt = st.text_input("Enter the Person's fnlwgt (19302.0-1226583.0")
    education_num = st.text_input("Education level (1-16)")
    capital_gain= st.text_input("The Person's Capital Gain (0.0-99999.0) ")
    capital_loss = st.text_input("The Person's Capital Loss (0.0 -4356.0)")
    hours_per_week = st.text_input("Weekly Working Hours (1.0 - 99.0)")
    
    
    
    ## Numeric conversion
    # Convert inputs to numeric using pd.to_numeric or float conversion
    age = pd.to_numeric(age, errors='coerce') # errors ='coerce' - tells pandas to force any non-convertible values like text or invalid numbers to NAN
    fnlwgt = pd.to_numeric(fnlwgt, errors='coerce')
    education_num = pd.to_numeric(education_num, errors='coerce')
    capital_gain = pd.to_numeric(capital_gain, errors='coerce')
    capital_loss = pd.to_numeric(capital_loss, errors='coerce')    
    hours_per_week = pd.to_numeric(hours_per_week, errors='coerce')
    

    # code for prediction ### refer to prediction function
    income_level = '' 
    
    # creating  a prediction button
    if st.button("PREDICT"):
        income_level = income_evaluation_prediction([age,fnlwgt,education_num,capital_gain,capital_loss,hours_per_week])
    st.success(income_level)
    
 
if __name__ == '__main__':
    main()


    
    
    
    
    
    
    
    
    
    
    
    
