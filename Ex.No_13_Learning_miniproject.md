# Ex.No: 13 Learning – Use Supervised Learning  
### DATE:                                                                            
### REGISTER NUMBER : 212222060197
### AIM: 
To write a program to train the classifier for Wine Quality Prediction.
###  Algorithm:
Step 1: Start the program.

Step 2: Import necessary libraries:

pandas for data handling.
sklearn for model creation and dataset splitting.
pickle for model persistence.

Step 3: Load the wine quality dataset from a CSV file.

Step 4: Define the features and target variable:

Set X as the dataset features (excluding "quality" and any unique identifiers).
Set y as the target variable, which is "quality".

Step 5: Split the dataset into training and testing sets to evaluate model performance.

Step 6: Train a RandomForestClassifier on the training data using X_train and y_train.

Step 7: Save the trained model as a .pkl file using pickle for future use.

Step 8: Create a Streamlit web app:

Load the saved model file.
Define input fields for each wine feature using st.number_input for user input.

Step 9: Capture user inputs for all wine quality features:

For each feature (e.g., "Fixed acidity," "Volatile acidity"), take user inputs within appropriate ranges.

Step 10: Define a prediction button:

When pressed, prepare user data in the required format.
Use the loaded model to predict wine quality based on the input features.

Step 11: Display the predicted wine quality to the user:

If successful, show the result.
If an error occurs, display an appropriate error message.

Step 12: Stop the program.
### Program:
```
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # or any other model
from sklearn.model_selection import train_test_split
import pickle

# Load your dataset
data = pd.read_csv('WineQT.csv')  # Update with your dataset path

# Define features and target
X = data.drop(columns=['quality', 'Id'])  # Features
y = data['quality']  # Target variable

# Split the dataset (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()  # or any other model
model.fit(X_train, y_train)

# Save the model
with open('wine_quality_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```
```
import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model
model_path = 'wine_quality_model.pkl'  # Update this path to your model file
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Define the feature names used during training (without 'quality')
feature_names = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

# Streamlit app title
st.title("Wine Quality Predictor")

# Create input fields for each parameter
fixed_acidity = st.number_input("Fixed acidity", min_value=0.0)
volatile_acidity = st.number_input("Volatile acidity", min_value=0.0)
citric_acid = st.number_input("Citric acid", min_value=0.0)
residual_sugar = st.number_input("Residual sugar", min_value=0.0)
chlorides = st.number_input("Chlorides", min_value=0.0)
free_sulfur_dioxide = st.number_input("Free sulfur dioxide", min_value=0.0)
total_sulfur_dioxide = st.number_input("Total sulfur dioxide", min_value=0.0)
density = st.number_input("Density", min_value=0.0)
pH = st.number_input("pH", min_value=0.0)
sulphates = st.number_input("Sulphates", min_value=0.0)
alcohol = st.number_input("Alcohol", min_value=0.0)

# Create a button to make the prediction
if st.button("Predict Wine Quality"):
    # Prepare input data for prediction as a DataFrame with correct feature names
    user_data = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                                free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]],
                              columns=feature_names)

    # Make prediction
    try:
        predicted_quality = model.predict(user_data)

        # Display the result
        st.success(f"Predicted wine quality: {predicted_quality[0]}")
    except ValueError as e:
        st.error(f"Error in prediction: {str(e)}")
```
### Output:



![Screenshot 2024-11-07 230858](https://github.com/user-attachments/assets/2ebd3a29-77ee-4250-84eb-f02b4f7da278)




### Result:
Thus the system was trained successfully and the prediction was carried out.
