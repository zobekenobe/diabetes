import streamlit as st
import numpy as np
import pandas as pd

import pickle

with open('diabetes.pkl', 'rb') as f:
    data = pickle.load(f)

scaler = data[0]
model  = data[1]

st.title("Diabetes Detector using Machine Learning")

st.write("""This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes.
Content

Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

    Pregnancies: Number of times pregnant
    Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
    BloodPressure: Diastolic blood pressure (mm Hg)
    SkinThickness: Triceps skin fold thickness (mm)
    Insulin: 2-Hour serum insulin (mu U/ml)
    BMI: Body mass index (weight in kg/(height in m)^2)
    DiabetesPedigreeFunction: Diabetes pedigree function
    Age: Age (years)
    Outcome: Class variable (0 or 1)

Sources:

(a) Original owners: National Institute of Diabetes and Digestive and
Kidney Diseases
(b) Donor of database: Vincent Sigillito (vgs@aplcen.apl.jhu.edu)
Research Center, RMI Group Leader
Applied Physics Laboratory
The Johns Hopkins University
Johns Hopkins Road
Laurel, MD 20707
(301) 953-6231
(c) Date received: 9 May 1990""")

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

