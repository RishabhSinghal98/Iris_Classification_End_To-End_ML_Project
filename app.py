import streamlit as st
import pickle 
import pandas as pd

#reading the encoder, model and scaler object files
encoder = pickle.load(open("encoder.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))
scaler = pickle.load(open("scaler.pkl", 'rb'))

#setting the title and text
st.title("Iris Flower Classification 💐")
st.write("Designed by Rishabh")

html_temp = """
<style>
.css-18ni7ap{
    background: rgb(2,0,36);
background: linear-gradient(90deg, rgba(2,0,36,1) 0%, rgba(121,9,117,1) 2%, rgba(0,212,255,1) 83%);
    
}
.appview-container {
    background: rgb(2,0,36);
background: linear-gradient(90deg, rgba(2,0,36,1) 0%, rgba(121,9,117,1) 2%, rgba(0,212,255,1) 83%);
}
h1{
    color:white;
}
p{
    color:darkturquoise;
}
.css-16idsys p {
 color:white;

    
</style>
"""

st.markdown(html_temp,unsafe_allow_html=True)

#taking the input from user
Sepal_length = st.number_input("Enter Sepal Length (cm):", min_value=0.0)
Sepal_width = st.number_input("Enter Sepal Width (cm):", min_value=0.0)
Petal_Length = st.number_input("Enter Petal Length (cm):", min_value=0.0)
Petal_width = st.number_input("Enter Petal Width (cm):", min_value=0.0)

#button to trigger the classification
if st.button("Predict"):
    newValue = pd.DataFrame([[Sepal_length, Sepal_width, Petal_Length, Petal_width]])
    newValue = scaler.transform(newValue)
    prediction = model.predict(newValue)
    finalAns = encoder.inverse_transform(prediction)
    st.markdown(f"The Specie Is **{finalAns[0]}**")