import streamlit as st
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

soilclass=pickle.load(open('soil.pkl','rb'))
crop_p=pickle.load(open('crop.pkl','rb'))
fert_r=pickle.load(open('fertilizer.pkl','rb'))

def soil_classification(n,p,k,ph): 
    input=np.array([[n,p,k,ph]]).astype(np.float64)
    prediction=soilclass.predict(input)
    pred= (f"{prediction.item().title()}")
    return str(pred)

def crop_prediction(temp,humid,rain,soil):
    input=np.array([[temp,humid,rain,soil]]).astype(np.float64)
    prediction=crop_p.predict(input)
    pred= (f"{prediction.item().title()}")
    return str(pred)

def fert_recomd(n,p,k,cr):
    input=np.array([[n,p,k,cr]]).astype(np.object)
    prediction=fert_r.predict(input)
    pred= (f"{prediction.item().title()}")
    return str(pred)


def soil_explore(df):
    
    if st.checkbox('Dataset view'):
        st.subheader('Soil Dataset')
        st.write(df)
    if st.checkbox('Shape of Data'):
        st.subheader('Shape of Data')
        shape= df.shape
        st.write(shape)
    if st.checkbox('Datatype of attribute'):
        st.subheader('Datatype of attribute')
        d= df.dtypes
        st.write(d)
    if st.checkbox('Correlation graph'):
        st.subheader('Correlation graph')
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = df.corr()
        sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
        square=True, ax=ax)
        st.write(fig)

def main():

    html_temp = """
        <div style="background-color:#025246 ;padding:10px">
        <h2 style="color:white;text-align:center;">Soil Classfication, Crop Prediction and Fertilizer Recommendation using Deep Learning Approach </h2>
        </div>
        <div style=background-image: url(C://Users/AISHWARYA SKULKARNI/streamlit_app/demo/banner.png); height: 200px; width: 400px; border: 1px solid black;">
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)
    


    select_op=st.sidebar.selectbox('Select option', 
                                    ('<Select>','Home', 'Explore Data', 'Service'))
    
    if select_op == 'Home':
        st.subheader("About")
        st.write(
            "In our country, agriculture is a major source of food production to the growing demand of the human population. The contribution of Indian agriculture sector to India’s Gross Domestic Product (GDP) is about 18% and which in average provides employment to 50% of the country’s workforce, which clearly shows that agriculture plays a vital role in the gross economy. Agriculture is a major contributor to the Indian economy. It is very important to monitor the soil nutrients for a good yield. Agriculture sector faces many problems such as irregular rainfall, floods, draught, climate change etc. To overcome these problems technological solution is needed which can help the farmers. The productivity of farming is not only depending on natural resources but it also depends on input provided to the system. "
        )
        st.write(
            "The decision of a farmer regarding which type of crop to grow in his land is generally depends on his intuition and many other factors such as making huge profits within a short period of time, lack of awareness about the demand in the market and when he overestimates a soil’s potential to support the growth of a particular type of crop and many more. A wrong decision that is taken on the farmer's side could put a much bigger pressure on the financial condition of his family resulting in severe loss. For all this reason we can see a farmer’s pressure regarding which crop should be grown in his land. So now the most important aspect is to design a recommendation system that predicts the type of crop that can be grown in a particular land and thereby helping the farmers. With this aim in mind, we have decided to develop a system that takes in the soil parameters like N, P, K (Nitrogen, Phosphorus, Potassium) and the pH values and predicts the most suitable crop that can be grown in that region."
        )
        st.write(
            "Deep learning is used worldwide these days with great efficiency in sectors such as forecasting, pattern recognition, fraud/fault detection, prediction, virtual assistance, image processing, robotics, artificial intelligence and so on and so forth. The agricultural sector has work done on it through weather forecasting, crop disease prediction, yield prediction, etc. However, what we propose, which is soil classification and crop prediction and crop recommendation itself, has not yet been implemented on a mass scale over large datasets and establishing a relationship among them to solve the problem through data analysis. "
        )
        st.subheader("Usage")
        st.write(
            "1. Explore Soil and Fertilizer Dataset \n"
            "2. Soil Classification \n"
            "3. Crop Prediction \n"
            "4. Fertilizer Recommendation \n"
        )
    elif select_op == 'Explore Data':
        st.subheader("Explore Soil and Fertilizer Dataset")
        select_db=st.selectbox('Select dataset', 
                                    ('<Select>','Soil Dataset', 'Fertilizer Dataset'))
        if select_db == 'Soil Dataset':
            st.text('You have selected Soil Dataset')
            # Reading soil and crop dataset
            df = pd.read_csv(r'soil.csv')
            if st.checkbox('Attributes of Soil data'):
                st.subheader('Attributes Description')
                st.write(
                    "1. N - ratio of Nitrogen content in soil  \n"
                    "2. P - ratio of Phosphorous content in soil  \n"
                    "3. K - ratio of Potassium content in soil  \n"
                    "4. temperature - temperature in degree Celsius  \n"
                    "5. humidity - relative humidity in %ph - ph value of the soil  \n"
                    "6. rainfall - rainfall in mm  \n"
                )
            soil_explore(df) 
        elif select_db == 'Fertilizer Dataset':
            st.text('You have selected Fertilizer Dataset')
            df = pd.read_csv(r'Fertilizer Prediction.csv')
            if st.checkbox('Attributes of Fertilizer data'):
                st.subheader('Attributes Description')
                st.write(
                    "1. N - ratio of Nitrogen content in soil  \n"
                    "2. P - ratio of Phosphorous content in soil  \n"
                    "3. K - ratio of Potassium content in soil  \n"
                    "4. temperature - temperature in degree Celsius  \n"
                    "5. mostuire - Ratio of the mass of water \n"
                    "6. humidity - relative humidity in %ph - ph value of the soil  \n"
                    "7. crop - names of different crop \n"
                    "8. Fertilizer - names of different fertilizer \n"
                )
            soil_explore(df)
    elif select_op == 'Service':
        n = st.text_input('Nitrogen')
        p = st.text_input('Phosphorous')
        k = st.text_input('Potassium')
        ph = st.text_input('PH')
        temp = st.text_input('Temperature')
        humid = st.text_input('Humidity')
        rain = st.text_input('Rainfall')
        output=soil_classification(n,p,k,ph)

        if st.checkbox('Soil Nature'):
            st.success('Soil nature for given soil nutrients is: {}'.format(output))

        if st.checkbox('Crop'):
            if output == 'Ultra Acidic':
                crop_op=crop_prediction(temp,humid,rain,"1")
                st.success('Crop is: {}'.format(crop_op))
            elif output == 'Extremely Acidic':
                crop_op=crop_prediction(temp,humid,rain,"2")
                st.success('Crop is: {}'.format(crop_op))
            elif output == 'Very strongly Acidic':
                crop_op=crop_prediction(temp,humid,rain,"3")
                st.success('Crop is: {}'.format(crop_op))
            elif output == 'Strongly Acidic':
                crop_op=crop_prediction(temp,humid,rain,"4")
                st.success('Crop is: {}'.format(crop_op))
            elif output == 'Moderately Acidic':
                crop_op=crop_prediction(temp,humid,rain,"5")
                st.success('Crop is: {}'.format(crop_op))
            elif output == 'Slightly Acidic':
                crop_op=crop_prediction(temp,humid,rain,"6")
                st.success('Crop is: {}'.format(crop_op))
            elif output == 'Neutral':
                crop_op=crop_prediction(temp,humid,rain,"7")
                st.success('Crop is: {}'.format(crop_op))
            elif output == 'Slightly Alkaline' :
                crop_op=crop_prediction(temp,humid,rain,'8')
                st.success('Crop is: {}'.format(crop_op))
            elif output == 'Moderately Alkaline':
                crop_op=crop_prediction(temp,humid,rain,"9")
                st.success('Crop is: {}'.format(crop_op))
            elif output == 'Strongly Alkaline':
                crop_op=crop_prediction(temp,humid,rain,"10")
                st.success('Crop is: {}'.format(crop_op))
            elif output == 'Very strongly Alkaline':
                crop_op=crop_prediction(temp,humid,rain,"11")
                st.success('Crop is: {}'.format(crop_op))
        
        if st.checkbox('Fertilizer') :
            if crop_op == 'apple' :
                fert_op=fert_recomd(n,p,k,"1")
                st.success('Crop is: {}'.format(fert_op))
            elif crop_op == 'banana' :
                fert_op=fert_recomd(n,p,k,"2")
                st.success('Crop is: {}'.format(fert_op))
            elif crop_op == 'chickpeas' :
                fert_op=fert_recomd(n,p,k,"3")
                st.success('Crop is: {}'.format(fert_op))
            elif crop_op == 'coconut' :
                fert_op=fert_recomd(n,p,k,"4")
                st.success('Crop is: {}'.format(fert_op))
            elif crop_op == 'coffee' :
                fert_op=fert_recomd(n,p,k,"5")
                st.success('Crop is: {}'.format(fert_op))
            elif crop_op == 'cotton' :
                fert_op=fert_recomd(n,p,k,"6")
                st.success('Crop is: {}'.format(fert_op))
            elif crop_op == 'grapes' :
                fert_op=fert_recomd(n,p,k,"7")
                st.success('Crop is: {}'.format(fert_op))
            elif crop_op == 'jute' :
                fert_op=fert_recomd(n,p,k,"8")
                st.success('Crop is: {}'.format(fert_op))
            elif crop_op == 'lentil' :
                fert_op=fert_recomd(n,p,k,"9")
                st.success('Crop is: {}'.format(fert_op))
            elif crop_op == 'maize' :
                fert_op=fert_recomd(n,p,k,"10")
                st.success('Crop is: {}'.format(fert_op))
            elif crop_op == 'mango' :
                fert_op=fert_recomd(n,p,k,"11")
                st.success('Crop is: {}'.format(fert_op))
            elif crop_op == 'pomegranate' :
                fert_op=fert_recomd(n,p,k,"12")
                st.success('Crop is: {}'.format(fert_op))
            elif crop_op == 'rice' :
                fert_op=fert_recomd(n,p,k,"13")
                st.success('Crop is: {}'.format(fert_op))
            elif crop_op == 'kidneybeans'or 'pigeonpeas' or 'mothbeans' or 'mungbean' or 'blackgram':
                fert_op=fert_recomd(n,p,k,"0")
                st.success('Crop is: {}'.format(fert_op))
            
if __name__=='__main__':
    main()