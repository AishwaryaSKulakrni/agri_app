import streamlit as st
import pickle
import numpy as np

soilmodel=pickle.load(open('soilclass.pkl','rb'))
crop=pickle.load(open('crop.pkl','rb'))

def soil_classification(n,p,k,ph):
    input=np.array([[n,p,k,ph]]).astype(np.float64)
    prediction=soilmodel.predict(input)
    pred= (f"{prediction.item().title()}")
    return str(pred)

def crop_prediction(temp,humid,rain):
    input=np.array([[temp,humid,rain]]).astype(np.float64)
    prediction=crop.predict(input)
    pred= (f"{prediction.item().title()}")
    return str(pred)

def main():
    st.title("Soil Classfication")

    n = st.text_input('Nitrogen')
    p = st.text_input('Phosphorous')
    k = st.text_input('Potassium')
    ph = st.text_input('PH')
    temp = st.text_input('Temperature')
    humid = st.text_input('Humidity')
    rain = st.text_input('Rainfall')

    output=soil_classification(n,p,k,ph)
    crop_op=crop_prediction(temp,humid,rain)

    if st.button("SUBMIT"):
        st.success('Soil nature for given soil nutrients is: {}'.format(output))
        st.success('Crop grown for given soil nutrients is: {}'.format(crop_op))

if __name__=='__main__':
    main()