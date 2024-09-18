#import streamlit for quick webapp building
import streamlit as st
import pandas as pd
import os

#import pandas for profiling
from ydata_profiling import ProfileReport

#import ML PyCaret librairies
from pycaret.classification import load_model as clas_load_model, predict_model as clas_predict_model, setup as clas_setup, compare_models as clas_compare_models, pull as clas_pull, save_model as clas_save_model
from pycaret.regression import load_model as reg_load_model, predict_model as reg_predict_model,setup as reg_setup, compare_models as reg_compare_models, pull as reg_pull, save_model as reg_save_model


with st.sidebar:
    st.image("https://hydrationfitness.com/wp-content/uploads/2014/03/foto-hydrat-brainwiring.jpg")
    st.title("Silas' Automated ML Webapp")
    choice = st.radio("Navigation", ["Upload","Profiling","ML","Download"])
    st.info("This application provides an automated series of tools to quicky train a ML model on your uploaded dataset. I use the following tools: Streamlit, Pandas Profiling, and PyCaret.\t Your Welcome :)")


if os.path.exists("sourcedata.csv"):
    dataFrame = pd.read_csv("sourcedata.csv", index_col=None)
    

if choice == "Upload":
    st.title("Upload Your Data!")
    file = st.file_uploader("Upload your dataset csv here")
    if file:
        #dataFrame = pd.read_csv(file, index_col= None)
        dataFrame.to_csv("sourcedata.csv", index = None)
        st.dataframe(dataFrame)
elif choice == "Profiling":
    if dataFrame is not None:
        st.title("Automated Profile Report")
        profile = ProfileReport(dataFrame)

        profile.to_file("profile_report.html")

        st.components.v1.html(open("profile_report.html", 'r').read(),height = 1000,scrolling = True)
    else:
        st.write("Please upload some data first")
elif choice == "ML":
    if dataFrame is not None:
        st.title("Machine Learning is cool!")
        target = st.selectbox("Select Your Target (# corrisponds to col #)", dataFrame.columns)

        if st.button("Train Model (With Regression **currently bugged)"):
            # Convert all columns to numeric, coercing errors
            dataFrame = dataFrame.apply(pd.to_numeric, errors='coerce')
            dataFrame = dataFrame.fillna(0)  

            
            reg_setup(dataFrame, target=target)
            setup_df = reg_pull()
            st.write("This is the ML Experiment settings")
            st.dataframe(setup_df)

            best_model = reg_compare_models()
            compare_df = reg_pull()
            st.write("This is the ML Model")
            st.dataframe(compare_df)
            best_model
            reg_save_model(best_model, 'best_model')


        if st.button("Train Model (With Classification)"):
            clas_setup(dataFrame, target = target)
            setup_df = clas_pull()
            st.write("This is the ML Experiment settings")
            st.dataframe(setup_df)

            best_model = clas_compare_models()
            compare_df = clas_pull()
            st.write("This is the ML Model")
            st.dataframe(compare_df)
            best_model
            clas_save_model(best_model,'best_model')

        # New section for prediction
        st.write("### Predict on a New Dataset")
        
        # Upload new dataset for prediction
        pred_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])
        
        if pred_file is not None:
            # Load the new data
            pred_data = pd.read_csv(pred_file)
            st.dataframe(pred_data)
            
            # Load the saved model
            model = clas_load_model('best_model')
            
            # Predict on new data
            predictions = clas_predict_model(model, data=pred_data)
            
            # Display the predictions
            st.write("Predictions:")
            st.dataframe(predictions)

elif choice == "Download":
    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", 'rb') as file:
            st.download_button("Download the model", file, "trained_model.pkl")