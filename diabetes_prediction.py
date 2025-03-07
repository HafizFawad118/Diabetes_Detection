import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the dataset
data = pd.read_csv('diabetes.csv')  # Make sure the file path is correct

# Remove the 'Pregnancies' column
data = data.drop('Pregnancies', axis=1)

# Features (X) and Target (y)
X = data.drop('Outcome', axis=1)  # Features (all columns except 'Outcome')
y = data['Outcome']  # Target (Outcome column)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Support Vector Machine (SVM) model with RBF kernel for better accuracy
model = SVC(kernel='rbf', C=1.0, gamma='scale')

# Train the model
model.fit(X_train, y_train)

# Streamlit app
st.title('Diabetes Prediction App')

# Calculate age-wise average values from the dataset
age_wise_avg = data.groupby('Age').mean().reset_index()

# Sidebar for user input (manual input)
st.sidebar.header('User Input Features')

# Function to get user input from text fields
def user_input_features():
    glucose = st.sidebar.text_input('Glucose', '0')
    blood_pressure = st.sidebar.text_input('Blood Pressure', '0')
    skin_thickness = st.sidebar.text_input('Skin Thickness', '0')
    insulin = st.sidebar.text_input('Insulin', '0')
    bmi = st.sidebar.text_input('BMI', '0.0')
    diabetes_pedigree_function = st.sidebar.text_input('Diabetes Pedigree Function', '0.0')
    age = st.sidebar.text_input('Age', '0')
    
    # Create a dictionary with correct data types
    data = {
        'Glucose': int(glucose),
        'BloodPressure': int(blood_pressure),
        'SkinThickness': int(skin_thickness),
        'Insulin': int(insulin),
        'BMI': float(bmi),  # Ensure BMI is float
        'DiabetesPedigreeFunction': float(diabetes_pedigree_function),  # Keep as float
        'Age': int(age)
    }
    
    # Convert to DataFrame
    features = pd.DataFrame(data, index=[0])
    
    # Ensure the column order matches the training data
    features = features[X.columns]
    return features

# Main section for file upload
st.header('File Upload Section')

# File uploader for CSV or Excel files
uploaded_file = st.file_uploader("Upload your input file (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read the uploaded file
    if uploaded_file.name.endswith('.csv'):
        input_df = pd.read_csv(uploaded_file)
    else:
        input_df = pd.read_excel(uploaded_file)
    
    # Drop 'Unnamed: 0' and 'Pregnancies' columns if they exist
    if 'Unnamed: 0' in input_df.columns:
        input_df = input_df.drop('Unnamed: 0', axis=1)
    if 'Pregnancies' in input_df.columns:
        input_df = input_df.drop('Pregnancies', axis=1)
    
    # Display the uploaded data in a clean table
    st.subheader('Uploaded File Data:')
    st.dataframe(input_df)  # Use st.dataframe for better alignment
    
    # Ensure the uploaded data has the same columns as the training data
    required_columns = X.columns
    if all(column in input_df.columns for column in required_columns):
        # Reorder columns to match training data
        input_df = input_df[required_columns]
        
        # Convert data types (except DiabetesPedigreeFunction)
        for col in input_df.columns:
            if col == 'BMI':
                input_df[col] = input_df[col].astype(float)  # Ensure BMI is float
            elif col != 'DiabetesPedigreeFunction':
                input_df[col] = input_df[col].astype(int)
        
        # Standardize the uploaded data
        input_scaled = scaler.transform(input_df)
        
        # Make predictions
        predictions = model.predict(input_scaled)
        
        # Add predictions to the DataFrame
        input_df['Prediction'] = predictions
        
        # Display normal values for each patient's age
        st.subheader('Normal Values for Each Patient:')
        for i, row in input_df.iterrows():
            patient_age = row['Age']
            normal_values = age_wise_avg[age_wise_avg['Age'] == patient_age].drop('Outcome', axis=1)
            
            # Ensure "Age" is the first column
            normal_values = normal_values[['Age'] + [col for col in normal_values.columns if col != 'Age']]
            
            # Convert data types (except DiabetesPedigreeFunction)
            for col in normal_values.columns:
                if col == 'BMI':
                    normal_values[col] = normal_values[col].astype(float)  # Ensure BMI is float
                elif col != 'DiabetesPedigreeFunction':
                    normal_values[col] = normal_values[col].astype(int)
            
            st.write(f"Patient {i+1} (Age: {patient_age}):")
            st.dataframe(normal_values.style.hide(axis='index'))
            
            if row['Prediction'] == 1:
                st.write("**This person has Diabetes.**")
            else:
                st.write("**This person does not have Diabetes.**")
            st.write("---")
    else:
        st.error(f"Uploaded file must contain the following columns: {required_columns}")

# Add a horizontal line to separate sections
st.write("---")

# Manual Input Section (replaced with a line)
st.header('Manual Input Section')

# Collect user input
input_df = user_input_features()

# Ensure "Age" is the first column in the user input
input_df = input_df[['Age'] + [col for col in input_df.columns if col != 'Age']]

# Display user input in a clean table
st.subheader('Patient Values:')
# Remove the index column and display the user input
st.dataframe(input_df.style.hide(axis='index'))

# Display normal values for the user's age
user_age = input_df['Age'].values[0]
if user_age > 0:
    normal_values = age_wise_avg[age_wise_avg['Age'] == user_age].drop('Outcome', axis=1)
    
    # Ensure "Age" is the first column in the normal values
    normal_values = normal_values[['Age'] + [col for col in normal_values.columns if col != 'Age']]
    
    # Convert data types (except DiabetesPedigreeFunction)
    for col in normal_values.columns:
        if col == 'BMI':
            normal_values[col] = normal_values[col].astype(float)  # Ensure BMI is float
        elif col != 'DiabetesPedigreeFunction':
            normal_values[col] = normal_values[col].astype(int)
    
    # Display normal values in a clean table
    st.subheader('Normal Values:')
    # Remove the index column and display the normal values
    st.dataframe(normal_values.style.hide(axis='index'))
else:
    st.write("**Please enter a valid age to see normal values.**")

# Make prediction
try:
    # Ensure the input_df has the correct column order before scaling
    input_df = input_df[X.columns]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    
    # Display result
    st.subheader('Prediction:')
    if user_age > 0:
        if prediction[0] == 1:
            st.write('**This person has Diabetes.**')
        else:
            st.write('**This person does not have Diabetes.**')
    else:
        st.write("**Please enter valid input to see the prediction.**")
except Exception as e:
    st.error(f"An error occurred during prediction: {e}")