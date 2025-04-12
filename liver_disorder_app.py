import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv("liver_data.csv.csv")

# Rename the label column for easier use (if it's named something odd)
df.rename(columns={df.columns[-1]: "Selector"}, inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)

# Encode gender to numeric
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # Male = 1, Female = 0

# Separate features and label
X = df.drop('Selector', axis=1)
y = df['Selector']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Streamlit App
st.title("🩺 Liver Disorder Predictor")

st.write("Enter your enzyme values and details below:")

# Widgets for user input
age = st.slider("Age", 1, 100, 45)
gender = st.radio("Gender", ["Male", "Female"])
tb = st.number_input("Total Bilirubin (TB)", min_value=0.0, max_value=75.0, value=1.0)
db = st.number_input("Direct Bilirubin (DB)", min_value=0.0, max_value=20.0, value=0.5)
alkphos = st.number_input("Alkaline Phosphotase", min_value=50, max_value=2100, value=200)
sgpt = st.number_input("SGPT", min_value=10, max_value=2000, value=30)
sgot = st.number_input("SGOT", min_value=10, max_value=2000, value=35)
tp = st.number_input("Total Proteins", min_value=0.0, max_value=10.0, value=6.5)
alb = st.number_input("Albumin", min_value=0.0, max_value=10.0, value=3.0)
ag_ratio = st.number_input("A/G Ratio", min_value=0.0, max_value=3.0, value=1.0)

# Convert gender to 0 or 1
gender_encoded = 1 if gender == "Male" else 0

# Prepare input
input_data = pd.DataFrame([[age, gender_encoded, tb, db, alkphos, sgpt, sgot, tp, alb, ag_ratio]],
                          columns=X.columns)

# Predict
if st.button("Predict Liver Condition"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.warning("⚠️ The model suggests this person **may have a liver disorder**.")
    else:
        st.success("✅ The model suggests this person is likely **healthy**.")

