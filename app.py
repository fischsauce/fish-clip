"""
This is a web app created with Streamlit to host this project. Feel free to use this file as a guide or visit my
article on the topic (linked below).
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from sklearn.linear_model import LogisticRegressionCV
import PIL.Image

import sys
sys.path.append("clipit")
sys.path.append("/clipit")
sys.path.append("CLIP")
sys.path.append("/CLIP")
sys.path.append("diffvg")
sys.path.append("/diffvg")
sys.path.append("taming-transformers")
sys.path.append("/taming-transformers")


st.header("ClipIt")
st.write("""x""")

st.sidebar.header('User Input')

# def user_input_features():
#     prompts = st.sidebar.text_input('prompts'),
#     data = {'prompts': prompts}
#     features = pd.DataFrame(data, index=[0])
#     return features


prompts = st.sidebar.text_input('prompts')


# input_df = user_input_features()

# df = pd.read_csv('https://query.data.world/s/fzhdybgova7pqh6amwfzrnhumdc26t')


use_pixeldraw = True #@param {type:"boolean"}

import clipit


clipit.reset_settings()

clipit.add_settings(prompts=prompts, size=[256, 256])

clipit.add_settings(quality="normal", iterations=10, pixel_scale=0.5)

clipit.add_settings(use_pixeldraw=use_pixeldraw)

settings = clipit.apply_settings()

clipit.do_init(settings)
clipit.do_run(settings)


image = Image.open('output.png') 

st.image(image, caption=prompts)








# Data Cleaning Steps
# df.drop_duplicates(subset='patient_nbr', inplace=True)
# df.drop(['encounter_id','patient_nbr','weight', 'payer_code', 'medical_specialty'], axis=1, inplace=True)
# df = df[df.race != '?'] # about 1,000 obs
# df = df[df.gender != 'Unknown/Invalid'] # 1 obs
# df.readmitted.replace({'NO': 0, '<30': 1, '>30': 2}, inplace=True)

# df = df[pd.to_numeric(df['diag_1'], errors='coerce').notnull()]
# df = df[pd.to_numeric(df['diag_2'], errors='coerce').notnull()]
# df = df[pd.to_numeric(df['diag_3'], errors='coerce').notnull()]

# df.diag_1 = df.diag_1.astype('float64')
# df.diag_2 = df.diag_2.astype('float64')
# df.diag_3 = df.diag_3.astype('float64')

# # Feature Engineering
# df['A1C_test'] = np.where(df.A1Cresult == 'None', 0, 1)
# df.change = np.where(df.change == 'No', 0, 1)
# df['A1C_test_and_changed'] = np.where((df.change == 1) & (df.A1C_test == 1), 1, 0)

# conditions = [
#     (df.age ==  '[0-10)') | (df.age == '[10-20)') | (df.age == '[20-30)'),
#     (df.age == '[30-40)') | (df.age == '[40-50)') | (df.age == '[50-60)'),
#     (df.age == '[60-70)') | (df.age == '[70-80)') | (df.age == '[80-90)') | (df.age == '[90-100')]

# choices = [
#     '[0-30)',
#     '[30-60]',
#     '[60-100)']

# df['binned_age'] = np.select(conditions, choices, default=np.nan)
# df = df[df.binned_age != 'nan']
# df.drop(['age'], axis=1, inplace=True)

# df['diabetes_as_diag_1'] = np.where((df.diag_1 >= 250) & (df.diag_1 <251), 1, 0)
# df['diabetes_as_diag_2'] = np.where((df.diag_2 >= 250) & (df.diag_2 <251), 1, 0)
# df['diabetes_as_diag_3'] = np.where((df.diag_3 >= 250) & (df.diag_3 <251), 1, 0)
# df.drop(['diag_1', 'diag_2', 'diag_3'], axis=1, inplace=True)

# meds_to_remove = ['repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'tolbutamide',
#             'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide',
#             'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
#             'metformin-rosiglitazone', 'metformin-pioglitazone']
# df.drop(meds_to_remove, axis=1, inplace=True)

# X = df.drop('readmitted', axis = 1)
# df = pd.concat([input_df, X], axis=0)

# encode = ['race', 'gender', 'max_glu_serum', 'A1Cresult', 'metformin', 'glipizide', 'glyburide',
#           'insulin', 'diabetesMed', 'binned_age']

# for col in encode:
#     dummy = pd.get_dummies(df[col], prefix=col)
#     df = pd.concat([df, dummy], axis=1)
#     del df[col]
# df = df[:1]

#Write out input selection
# st.subheader('User Input (Pandas DataFrame)')
# # st.write(df)

# #Load in model
# # load_clf = pickle.load(open('diabetes_model.pkl', 'rb'))

# # Apply model to make predictions
# # prediction = load_clf.predict(df)
# # prediction_proba = load_clf.predict_proba(df)

# st.subheader('Prediction')
# st.write("""
# This is a multi-class classification model. Options are: 
# 1) 'NO' --> this patient was not readmitted within a year, 
# 2) '<30' --> this patient was readmitted within 30 days, or 
# 3) '>30' --> this patient was readmitted after 30 days. 
# This generally corresponds to the severity of the patient's diabetes as well as the specific care, or lack thereof, during the visit.
# """)

# readmitted = np.array(['NO','<30','>30'])
# # st.write(readmitted[prediction])

# st.subheader('Prediction Probability')
# st.write("""
# 0 --> 'NO'
# 1 --> '<30'
# 2 --> '>30'
# """)
# # st.write(prediction_proba)

# st.subheader('Exploratory Data Analysis')
# st.write("""
# We identified some important features in the readmittance rate that you can explore below. To begin, here is the distribution
# of the classes in the original data set. We see that a majority of patients are not readmitted within a year. Patients that 
# are readmitted often have complications to their diabetes or the specific care recieved.
# """)
# st.image(Image.open('Images/Readmit_rate.png'), width = 500)

# st.write("""
# Now looking at the patient population given the long-term blood sugar HbA1c test, we see only about 20% of patients received
# this test, but, of those, 50% then had their medication changed and were less likely to be readmitted.
# """)
# st.image(Image.open('Images/HbA1c_test.png'), width = 500)

# st.write("""
# Finally, we see that age plays an important role. As expected, older patients have more complications due to their diabetes.
# Age was binned according to this chart into 0-30, 30-60, and 60-100.
# """)
# st.image(Image.open('Images/Readmit_vs_age.png'), width = 500)

# st.subheader('More Information')
# st.write("""
# For a deeper dive into the project, please visit the [repo on GitHub](https://github.com/ArenCarpenter/Diabetes_Hospitalizations) 
# where you can find all the code used in analysis, modeling, visualizations, etc. You can also read my 
# [articles](https://arencarpenter.medium.com/) in Towards Data Science on my other projects. 
# """)