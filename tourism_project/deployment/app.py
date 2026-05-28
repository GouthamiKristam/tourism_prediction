import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download, login
import joblib
import os

# Explicitly log in to Hugging Face Hub (required for hf_hub_download)
# Assuming HF_TOKEN is set as an environment variable in the deployment environment
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    st.error("HF_TOKEN environment variable not set. Cannot download model.")
    st.stop() # Stop execution if token is missing

# Download and load the trained model
repo_id = "GouthamiKristam/tourism-prediction-model" # Updated repository ID
filename = "best_tourism_model_v1.joblib" # Updated model filename
model_path = hf_hub_download(repo_id=repo_id, filename=filename)
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Package Purchase Prediction") # Updated title
st.write("""
This application predicts the likelihood of a customer purchasing a **Wellness Tourism Package**
based on various demographic and interaction characteristics.
Please enter the customer details below to get a purchase likelihood prediction.
""") # Updated description

# Define the category mappings based on LabelEncoder's typical alphabetical order
# These mappings are crucial to match the format the model was trained on.
# If the actual mappings differ from alphabetical, the model's prediction might be inaccurate.
typeof_contact_map = {"Company Invited": 0, "Self Inquiry": 1}
occupation_map = {"Freelancer": 0, "Large Business": 1, "Salaried": 2, "Small Business": 3, "Unemployed": 4}
gender_map = {"Female": 0, "Male": 1}
product_pitched_map = {"Basic": 0, "Deluxe": 1, "King": 2, "Standard": 3, "Super Deluxe": 4}
marital_status_map = {"Divorced": 0, "Married": 1, "Single": 2}
designation_map = {"AVP": 0, "Executive": 1, "Manager": 2, "President": 3, "Senior Manager": 4, "VP": 5}

# User input fields for numeric features
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=120, value=10, step=1)
number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2, step=1)
number_of_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=3, step=1)
preferred_property_star = st.number_input("Preferred Property Star (1-5)", min_value=1, max_value=5, value=3, step=1)
number_of_trips = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=5, step=1)
passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3, step=1)
own_car = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0, step=1)
monthly_income = st.number_input("Monthly Income", min_value=0.0, max_value=100000.0, value=5000.0, step=100.0)

# User input fields for categorical features (using selectbox and then mapping to integers)
typeof_contact_str = st.selectbox("Type of Contact", list(typeof_contact_map.keys()))
occupation_str = st.selectbox("Occupation", list(occupation_map.keys()))
gender_str = st.selectbox("Gender", list(gender_map.keys()))
product_pitched_str = st.selectbox("Product Pitched", list(product_pitched_map.keys()))
marital_status_str = st.selectbox("Marital Status", list(marital_status_map.keys()))
designation_str = st.selectbox("Designation", list(designation_map.keys()))

# Map string inputs to their corresponding label-encoded integers
typeof_contact = typeof_contact_map.get(typeof_contact_str)
occupation = occupation_map.get(occupation_str)
gender = gender_map.get(gender_str)
product_pitched = product_pitched_map.get(product_pitched_str)
marital_status = marital_status_map.get(marital_status_str)
designation = designation_map.get(designation_str)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'NumberOfFollowups': number_of_followups,
    'PreferredPropertyStar': preferred_property_star,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'MonthlyIncome': monthly_income,
    'TypeofContact': typeof_contact,
    'Occupation': occupation,
    'Gender': gender,
    'ProductPitched': product_pitched,
    'MaritalStatus': marital_status,
    'Designation': designation
}])

# Predict button
if st.button("Predict Purchase Likelihood"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    # Assuming the regressor output can be interpreted as a likelihood/probability
    st.success(f"Estimated Purchase Likelihood: **{prediction*100:.2f}%**")
