import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title('Penguin Classifier')
st.write('This app uses 6 inputs to classify penguins usng the Palmers Penguins dataset')

penguin_file = st.file_uploader('Upload your own penguin data')

if penguin_file is None:
    rf_pickle = open('random_forest_penguin.pickle', 'rb')
    map_pickle = open('output_penguin.pickle', 'rb')
    rfc = pickle.load(rf_pickle)
    unique_penguin_mapping = pickle.load(map_pickle)
    rf_pickle.close()
    map_pickle.close()
else:
    penguin_df = pd.read_csv(penguin_file)
    penguin_df.dropna()
    output = penguin_df['species']
    features = penguins_df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
    features = pd.get_dummies(features)
    output, uniques = pd.factorize(output)
    X_train, X_test, y_train, y_test = train_test_split(features, output, test_size=.8, stratify=output, random_state=8)
    rfc = RandomForestClassifier(random_state=8)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    score = round(accuracy_score(y_pred, y_test), 2)
    st.write('We trained a Random Forest Model on these data and the accuracy score is {}'.format(score))
    
with st.form('User inputs'):
    island = st.selectbox('Penguin Island', options=['Biscoe', 'Dream', 'Torgerson'])
    sex = st.selectbox('Sex', options=['Male', 'Female'])
    bill_length = st.number_input('Bill Length (mm)', min_value=0)
    bill_depth = st.number_input('Bill Depth (mm)', min_value=0)
    flipper_length = st.number_input('Flipper Length (mm)', min_value=0)
    body_mass = st.number_input('Body mass (g)', min_value=0)
    
    st.form_submit_button()

island_biscoe, island_dream, island_Torgerson = 0, 0, 0
if island == 'Biscoe':
    island_biscoe = 1
elif island == 'Dream':
    island_dream = 1
elif island == 'Torgerson':
    island_Torgerson = 1
    
sex_male, sex_female = 0, 0
if sex == 'Male':
    sex_male = 1
elif sex == 'Female':
    sex_female = 1
    
new_prediction = rfc.predict([[bill_length, bill_depth, flipper_length, body_mass, island_biscoe, island_dream, island_Torgerson, sex_male, sex_female]])

prediction_species = unique_penguin_mapping[new_prediction][0]

st.write('We predict your penguin is of species {}'.format(prediction_species))

st.write('We used a random forest model predict the species. The feature importances are below')
st.image('feature_importance.png')