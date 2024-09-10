# Importing necessary packages
import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(page_title='Machine Learning Apps')
st.title("Machine Learning Apps")
st.subheader("Random Forest")
st.write(":blue[Created by : YB Aditya]")
st.markdown("---")

st.caption("""Random Forest is an ensemble learning method primarily used for classification and regression tasks. It is based on the idea of creating multiple decision trees and then combining their predictions to form a single prediction.""")

image = Image.open('random_forest.jpg')
st.image(image, caption='random forest')

chooice = st.selectbox("Select Machine Learning Model", ("Random Forest Regression", "Random Forest Classification"))

uploaded_file = st.file_uploader('Choose a XLSX file', type='xlsx')
st.caption("_please make sure there is no Null-data, null data will make machine learning error_")
if uploaded_file:
    st.markdown('---')
    data = pd.read_excel(uploaded_file, engine='openpyxl')
    st.write("**Data Preview :**")
    st.write(data.head())
    st.write("**Data Information :**")
    st.caption("""**count** : The number of not-empty values. 
                  ,**mean** : The average (mean) value.
                  ,**std** : The standard deviation.
                  ,**min** : the minimum value.
                  ,**25%** : The 25% percentile*.
                  ,**50%** : The 50% percentile*.
                  ,**75%** : The 75% percentile*.
                  ,**max** : the maximum value.""")
    describe = data.describe()
    st.write(describe)

    st.write("**Corelation Between Data :**")
    st.caption("_Note: **1** is Perfect corelation, **0,9-06** Good Corelation, **<0,5** Bad Corelation, use this data to select Independent Variables_")
    correlation = data.corr(numeric_only=True, method="pearson")
    st.write(correlation)

    st.write("**Select The Target Variable (_Y_):**")
    target_var = st.selectbox("Target Variable", list(data.columns))

    st.write("**Select The Independent Variables (_X1, X2, X3, ..._):**")
    independent_vars = st.multiselect("Independent Variables", list(data.columns))
    
    
def random_forest_regression():
    # Plot Graph Correlation
    st.write("**Graph Corelation Between Data :**")
    fig = sns.pairplot(data, x_vars=independent_vars, y_vars=independent_vars, hue=target_var)
    st.pyplot(fig)
    fig_1 = sns.pairplot(data, x_vars=independent_vars, y_vars=target_var, hue=target_var)
    st.pyplot(fig_1)
    # Train a Multiple Regression model and display the results
    X = data[independent_vars]
    y = data[target_var]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test, random_state=random)
    rfr = RandomForestRegressor(n_estimators=estimator)
    model = rfr.fit(x_train, y_train)
    y_pred = rfr.predict(x_test)

    st.write("**Regression Performance Evaluation**")
    reg_score = model.score(x_test, y_test)
    st.write("Regression Score :", reg_score )
    st.caption(f"Regression Score also mean model accuracy, your model accurancy is {reg_score*100}%")

    mse = mean_squared_error(y_test, y_pred)
    st.write("Mean Squared Error :", mse)
    mae = mean_absolute_error(y_test, y_pred)
    st.write("Mean Absolute Error :", mae)
    rmse = np.sqrt(mse)
    st.write("Root Mean Squared Error :", rmse)
      
    # Plot Data
    st.write("**Linear Graph Actual vs Prediction**")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, color='g')
    ax.plot(y_pred, y_pred, color='k' )
    st.pyplot(fig)

    st.write("**Do you want to download this model ?**")
    st.caption("**WARNING THIS WILL RELOAD THE PAGE!**")
    st.download_button(
    "Download Model",
    data=pickle.dumps(model),
    file_name="model.pkl",
    )

    predict_var = rfr.predict([variable])
    st.sidebar.write("Prediction Value :",predict_var[0])

    
def random_forest_classification():
    # Plot Graph Correlation
    st.write("**Graph Corelation Between Data :**")
    fig = sns.pairplot(data, x_vars=independent_vars, y_vars=independent_vars, hue=target_var)
    st.pyplot(fig)
    fig_1 = sns.pairplot(data, x_vars=independent_vars, y_vars=target_var, hue=target_var)
    st.pyplot(fig_1)
    # Train a Multiple Regression model and display the results
    X = data[independent_vars]
    y = data[target_var]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test, random_state=random)
    rfc = RandomForestClassifier(n_estimators=estimator)
    model = rfc.fit(x_train, y_train)
    y_pred = rfc.predict(x_test)

    st.write("**Performance Evaluation**")
    reg_score = model.score(x_test, y_test)
    st.write("Score :", reg_score )
    st.caption(f"Score also mean model accuracy, your model accurancy is {reg_score*100}%")

    con_matrix = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix :")
    st.dataframe(pd.DataFrame(con_matrix))
    st.caption("Confusion Matrix :  matrix shows the number of correct and incorrect predictions made by the classifier.")

    st.write("Classification Report :")
    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())
    st.caption("Classification Report :  report shows precision, recall, f1-score, support and accuracy metrics for each class in the dataset.")


    st.write("**Do you want to download this model ?**")
    st.caption("**WARNING THIS WILL RELOAD THE PAGE!**")
    st.download_button(
    "Download Model",
    data=pickle.dumps(model),
    file_name="model.pkl",
    )

    predict_var = rfc.predict([variable])
    st.sidebar.write("Prediction Value :",predict_var[0])
    
st.sidebar.write("_**Input Machine Learning Parameter**_")
test = st.sidebar.slider("Input test size",min_value=0.1, max_value=0.4, value=0.2)
st.sidebar.caption("define test size from the dataset. 0.2 mean 20% from total dataset")
random = st.sidebar.number_input("Input random state", min_value=0, max_value=50)
st.sidebar.caption("random state is a model hyperparameter used to control the randomness involved in machine learning models")    
estimator = st.sidebar.number_input("Input n_estimators", min_value=0, max_value=1000)    
st.sidebar.caption("n_estimators : This is the number of trees you want to build before taking the maximum voting or averages of predictions. Higher number of trees give you better performance but makes your code slower.")
st.sidebar.write("---")

if uploaded_file and independent_vars is not None:
    variable = []
    st.sidebar.write("_**Input Data for Prediction**_")
    for i in range(len(independent_vars)):        
        var=st.sidebar.number_input(f"Input {independent_vars[i]}")
        variable.append(var)

if chooice == "Random Forest Regression":
    st.sidebar.write("This button will run Random Forest Regression")
    ml_button_rgs = st.sidebar.button("Run Regression")
    st.sidebar.write("---")
    if ml_button_rgs:
        random_forest_regression()

if chooice == "Random Forest Classification":
    st.sidebar.write("This button will run Random Forest Classification")
    ml_button_cls = st.sidebar.button("Run Classification")
    st.sidebar.write("---")
    if ml_button_cls:
        random_forest_classification()
    


