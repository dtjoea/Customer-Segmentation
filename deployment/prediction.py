import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

#Load model

with open('list_num_cols.txt', 'r') as file_6:
  num_columns = json.load(file_6)

with open('model_scalerminmax.pkl', 'rb') as file_4:
  minmaxscale = pickle.load(file_4)

with open('model_scalerob.pkl', 'rb') as file_5:
  robscale = pickle.load(file_5)

with open('model_logreg.pkl', 'rb') as file_1:
  model_logreg = pickle.load(file_1)

def run():
    with st.form('form_credit_card'):
        name = st.text_input('Name', value = '')
        sex=st.selectbox('sex', ('Male', 'Female'), index=1)
        age = st.number_input('age', min_value = 21, max_value=70, value = 25, step = 1, help = 'Usia Pemilik kartu kredit')
        marital_status=st.selectbox('marital_status', ('1', '2'), index=1)
        education_level = st.number_input('education_level', min_value = 0, max_value=6, value = 6)
        pay_2= st.number_input('pay_2', min_value = -2, max_value=8, value = 8)
        limit_balance= st.number_input('Pace', min_value = 100000, max_value=800000, value = 100000)
        pay_0 = st.number_input('pay_0', min_value = -2, max_value=8, value = 0)
        pay_3 = st.number_input('pay_3', min_value = -2, max_value=8, value = 0)
        pay_4 = st.number_input('pay_4', min_value = -2, max_value=8, value = 0)
        pay_5 = st.number_input('pay_5', min_value = -2, max_value=8, value = 0)
        pay_6 = st.number_input('pay_6', min_value = -2, max_value=8, value = 0)
        bill_amt_1= st.number_input('bill_amt_1', min_value = -1000, max_value=800000, value = 50)
        bill_amt_2= st.number_input('bill_amt_2', min_value = -1000, max_value=800000, value = 50)
        bill_amt_3= st.number_input('bill_amt_3', min_value = -1000, max_value=800000, value = 50)
        bill_amt_4= st.number_input('bill_amt_4', min_value = -1000, max_value=800000, value = 50)
        bill_amt_5= st.number_input('bill_amt_5', min_value = -1000, max_value=800000, value = 50)
        bill_amt_6= st.number_input('bill_amt_6', min_value = -1000, max_value=800000, value = 50)
        pay_amt_1= st.number_input('pay_amt_1', min_value = 0, max_value=800000, value = 50)
        pay_amt_2= st.number_input('pay_amt_2', min_value = 0, max_value=800000, value = 50)
        pay_amt_3= st.number_input('pay_amt_3', min_value = 0, max_value=800000, value = 50)
        pay_amt_4= st.number_input('pay_amt_4', min_value = 0, max_value=800000, value = 50)
        pay_amt_5= st.number_input('pay_amt_5', min_value = 0, max_value=800000, value = 50)
        pay_amt_6= st.number_input('pay_amt_6', min_value = 0, max_value=800000, value = 50)

        #submit button
        submitted = st.form_submit_button('Predict')

    data_inf = {
        'name':name,
        'sex': sex,
        'age': age,
        'marital_status': marital_status,
        'education_level': education_level,
        'pay_2': pay_2,
        'limit_balance': limit_balance,
        'pay_0': pay_0,
        'pay_3': pay_3,
        'pay_4': pay_4,
        'pay_5': pay_5,
        'pay_6': pay_6,
        'bill_amt_1': bill_amt_1,
        'bill_amt_2': bill_amt_2,
        'bill_amt_3': bill_amt_3,
        'bill_amt_4': bill_amt_4,
        'bill_amt_5': bill_amt_5,
        'bill_amt_6': bill_amt_6,
        'pay_amt_1': pay_amt_1,
        'pay_amt_2': pay_amt_2,
        'pay_amt_3': pay_amt_3,
        'pay_amt_4': pay_amt_4,
        'pay_amt_5': pay_amt_5,
        'pay_amt_6': pay_amt_6
    }

    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    #jika tombol submit ditekan
    if submitted:
        #split between numerical columns
        data_inf_num = data_inf[num_columns]
        #feature scaling 
        data_inf_num_scaled = minmaxscale.transform(data_inf_num[['education_level','pay_2']])
        data_inf_num_scaled2 = robscale.transform(data_inf_num[['limit_balance','pay_0','pay_3','pay_4','pay_5','pay_6',
'bill_amt_1','bill_amt_2','bill_amt_3','bill_amt_4','bill_amt_5','bill_amt_6',
'pay_amt_1','pay_amt_2','pay_amt_3','pay_amt_4','pay_amt_5','pay_amt_6']])
        data_inf_final = np.concatenate([data_inf_num_scaled,data_inf_num_scaled2], axis = 1)
        #predict using logistic reg model
        y_pred_inf = model_logreg.predict(data_inf_final)
        st.write('# Default Payment Next Month: ', (int(y_pred_inf)))

if __name__ == '__main__':
    run()