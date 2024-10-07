import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import streamlit as st


# Sample data (replace with your actual data)
data = {'feature1': [1, 2, 3, 4, 5], 
        'feature2': [6, 7, 8, 9, 10], 
        'target': [0, 1, 0, 1, 0]}
df = pd.DataFrame(data)

# Separate features and target variable
X = df[['feature1', 'feature2']]
y = df['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and fit the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save the trained model and scaler
joblib.dump(model, 'logistic_model.pkl')
joblib.dump(scaler, 'scaler.pkl')


# --- Streamlit app (streamlit_app.py) ---
# Load the model and scaler
model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app layout
st.title('Logistic Regression Prediction App')

# Input form for features
st.sidebar.header('Input Features')
input_features = {}
for col in X.columns:
    input_features[col] = st.sidebar.number_input(col, 0.0)

# Create a DataFrame from the input features
input_df = pd.DataFrame([input_features])

# Standardize input features
input_df_scaled = scaler.transform(input_df)

# Make predictions
if st.button("Predict"):
    prediction = model.predict(input_df_scaled)

    # Display the prediction
    st.header('Prediction')
    st.write(prediction)