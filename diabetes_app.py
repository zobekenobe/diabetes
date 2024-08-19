import streamlit as st
import numpy as np
import pandas as pd

import pickle

with open('diabetes.pkl', 'rb') as f:
    data = pickle.load(f)

scaler = data[0]
model  = data[1]

st.title("Diabetes Detector using Machine Learning")

with st.form('myform'):
    pr = st.number_input(label = 'Pregnancies', min_value = 0, max_value = 10, value = 'min', step = 1)
    gu = st.number_input(label = 'Glucose mg/dL', min_value = 70, max_value = 199, step = 10, value = 'min')
    bp =  st.number_input(label = 'Diastolic Blood Pressure (mm Hg)', min_value = 50, max_value = 110)
    sk = st.number_input(label = 'Triceps Skin fold Thickness (mm)', min_value = 10, max_value = 50)
    si = st.number_input(label = "2-hour serum insulin (mu U/ml)", min_value = 5, max_value = 300)
    bm = st.number_input(label = 'BMI (weight in kg/(height in m)^2)', min_value = 10, max_value = 60)
    pe = st.number_input(label = "Diabetes pedigree function", min_value = 0.01, max_value = 2.5)  
    ag = st.number_input(label = 'age (years)', min_value = 21, max_value = 81)

    submitted = st.form_submit_button('Calculate')
    ypred = 0
    if submitted:
        x = [pr, gu, bp, sk, si, bm, pe, ag]
        xscaled = scaler.transform([x])
        ypred = model.predict(xscaled)

        if ypred == 1:
            st.write(f"The person/ patient is classified as :red[high risk] of having diabetes")
        elif ypred == 0:
            st.write(f"The person/ patient is classified as having a :green[low risk or absence] of diabetes")

