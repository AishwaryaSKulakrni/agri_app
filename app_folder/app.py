import streamlit as st
from multiapp import MultiApp
from app import home, explore # import your app modules here

app = MultiApp()

st.title("Soil Classification, Crop Prediction and Fertilizer recommendation using Deep Learning Apporach")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Data", explore.app)
# The main app
app.run()