import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

st.set_page_config(
    #bikin judul pagenya
    page_title= 'Credit Card Usage',
    #agar gaada padding
    layout= 'wide',
    #expand page
    initial_sidebar_state = 'expanded'
)

def run():

    #Membuat title
    st.title('Credit Card Evaluation')

    #membuat sub-header
    st.subheader('EDA untuk Analisa Dataset Credit Card')

    #menambahkan gambar
    image= Image.open('image.jpg')
    st.image(image, caption= 'Credit Card')

    #menambhkan deskripsi
    st.write('Page ini dibuat oleh David Tjoea')

    #membuat garis lurus
    st.markdown('---')

    #load dataframe
    df = pd.read_csv('P1G5_Set_1_david_tjoea.csv')
    st.dataframe(df)

    #membuat bar plot
    st.write('#### Plot Limit Balance User')
    fig = plt.figure(figsize=(15,5))
    sns.countplot(x='limit_balance',data=df)
    st.pyplot(fig)

    #membuat histogram
    st.write('#### Histogram of Age of Credit Card User')
    fig = plt.figure(figsize=(15,5))
    sns.histplot(df['age'],bins=30, kde=True)
    st.pyplot(fig)

   
if __name__ == '__main__':
    run()