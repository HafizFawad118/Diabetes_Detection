

## Features

1. **Manual Input**: Users can input patient data (e.g., Glucose, Blood Pressure, BMI, etc.) manually via a sidebar.
2. **File Upload**: Users can upload a CSV or Excel file containing multiple patient records for bulk predictions.
3. **Prediction Results**: The app displays whether a patient has diabetes or not.
4. **Normal Values**: For each patient, the app shows the average normal values for their age group based on the dataset.
5. **User-Friendly Interface**: Built with Streamlit, the app provides an intuitive and interactive interface.

---

## Dataset

The dataset used for training the model is the **Pima Indians Diabetes Dataset**, which contains the following features:

- **Glucose**: Plasma glucose concentration.
- **BloodPressure**: Diastolic blood pressure (mm Hg).
- **SkinThickness**: Triceps skinfold thickness (mm).
- **Insulin**: 2-Hour serum insulin (mu U/ml).
- **BMI**: Body mass index (weight in kg/(height in m)^2).
- **DiabetesPedigreeFunction**: Diabetes pedigree function.
- **Age**: Age in years.
- **Outcome**: Target variable (1 = diabetes, 0 = no diabetes).

The `Pregnancies` column was removed from the dataset as it was deemed irrelevant for this prediction task.

---

## Model

The model used is a **Support Vector Machine (SVM)** with an **RBF kernel**. The dataset was split into training and testing sets (80% training, 20% testing), and the features were standardized using **StandardScaler** before training.

---

## How to Use the App

1. **Manual Input**:
   - Enter patient details in the sidebar (e.g., Glucose, Blood Pressure, BMI, etc.).
   - Click "Predict" to see the result.
   - The app will display whether the patient has diabetes and show normal values for their age.

2. **File Upload**:
   - Upload a CSV or Excel file containing patient data.
   - Ensure the file has the following columns: `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, and `Age`.
   - The app will display predictions and normal values for each patient in the file.

---

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Diabetes_Detection.git
   cd Diabetes_Detection
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open your browser and go to `http://localhost:8501` to access the app.

---

## Requirements

The following Python libraries are required to run the app:
- `numpy`
- `pandas`
- `scikit-learn`
- `streamlit`

You can install them using:
```bash
pip install numpy pandas scikit-learn streamlit
```

---

## Deployment

The app can be deployed using:
1. **Streamlit Sharing**: Upload the app to GitHub and deploy it using [Streamlit Sharing](https://share.streamlit.io/).
2. **Heroku**: Follow the [Heroku deployment guide](https://devcenter.heroku.com/articles/getting-started-with-python) for Streamlit apps.

---

## Screenshots

![Manual Input](screenshots/manual_input.png)
*Manual Input Section*

![File Upload](screenshots/file_upload.png)
*File Upload Section*

---

## Future Improvements

- Add more machine learning models (e.g., Random Forest, Logistic Regression) for comparison.
- Include additional evaluation metrics (e.g., precision, recall, F1-score).
- Allow users to select different datasets for training.
- Add visualizations (e.g., feature importance, confusion matrix).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Dataset: [Pima Indians Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
- Libraries: Streamlit, Scikit-learn, Pandas, NumPy

---

Feel free to customize this `README.md` file further to suit your needs!
