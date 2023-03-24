import streamlit as st
import pandas as pd
import numpy as np
import pickle
from FocalLoss import FocalLoss
from OneVSRestClassifier import *

st.write("""
# Cancer Prediction App
""")

st.sidebar.header('User Input Features')

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    sample_col = input_df['samples'].copy()
    fin_df = input_df[['1552296_at', '1555112_a_at', '1556195_a_at', '1556761_at',
       '1557316_at', '1563483_at', '1569607_s_at', '1570165_at', '201418_s_at',
       '201471_s_at', '202382_s_at', '202619_s_at', '203616_at', '204294_at',
       '204320_at', '204415_at', '204540_at', '204580_at', '204621_s_at',
       '205033_s_at', '205713_s_at', '205774_at', '206030_at', '206771_at',
       '207341_at', '207502_at', '207804_s_at', '209278_s_at', '209469_at',
       '209485_s_at', '209550_at', '209644_x_at', '209906_at', '210106_at',
       '210867_at', '211494_s_at', '212681_at', '213997_at', '214464_at',
       '214481_at', '214774_x_at', '214961_at', '214983_at', '217546_at',
       '218858_at', '219059_s_at', '219267_at', '219669_at', '219737_s_at',
       '220116_at', '220266_s_at', '220351_at', '220464_at', '220494_s_at',
       '221577_x_at', '222326_at', '223754_at', '223949_at', '224013_s_at',
       '224279_s_at', '224435_at', '224458_at', '226132_s_at', '227140_at',
       '227180_at', '227388_at', '227554_at', '227943_at', '228155_at',
       '228716_at', '229693_at', '230351_at', '230595_at', '230788_at',
       '232710_at', '232945_at', '233386_at', '233514_x_at', '235759_at',
       '236398_s_at', '236399_at', '237106_at', '238044_at', '238880_at',
       '239046_at', '239272_at', '239474_at', '240156_at', '240421_x_at',
       '241837_at', '243146_at']].copy()


# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(fin_df)
else:
    st.write('Awaiting CSV file to be uploaded')

# Reads in saved classification model
load_clf = pickle.load(open('cancer_clf.pkl', 'rb'))

st.subheader('Prediction')

rep_target = {
    0: 'normal',
    1: 'Non_small_cell_lung_cancer',
    2: 'CRC',
    3: 'HCC',
    4: 'oral_cavity_cancer',
    5: 'ccRCC',
    6: 'tumoral_breast',
    7: 'AML',
    8: 'tumoral_urothelia',
    9: 'tumoral_prostate',
    10: 'tumoral_pancreas'
}

# Apply model to make predictions
if uploaded_file is not None:
    prediction = load_clf.predict(fin_df)
    pred = pd.DataFrame(prediction)
    prediction_proba = load_clf.predict_proba(fin_df)

    pred.replace(rep_target, inplace=True)
    st.write(pred)

    st.subheader('Prediction Probability')
    st.write(prediction_proba)
