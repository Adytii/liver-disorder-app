import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv("liver_data.csv.csv")
df.rename(columns={df.columns[-1]: "Selector"}, inplace=True)
df.dropna(inplace=True)

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # Male = 1, Female = 0

X = df.drop('Selector', axis=1)
y = df['Selector']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Normal healthy liver value ranges
st.title("🩺 Liver Disorder Predictor")

st.write("Enter your enzyme values and details below (normal healthy ranges in brackets):")

age = st.slider("Age", 1, 100, 45)
gender = st.radio("Gender", ["Male", "Female"])
tb = st.number_input("Total Bilirubin (TB) [0.1 - 1.2 mg/dL]", min_value=0.0, max_value=75.0, value=1.0)
db = st.number_input("Direct Bilirubin (DB) [0.0 - 0.3 mg/dL]", min_value=0.0, max_value=20.0, value=0.5)
alkphos = st.number_input("Alkaline Phosphotase [45 - 115 U/L]", min_value=50, max_value=2100, value=200)
sgpt = st.number_input("SGPT [7 - 56 U/L]", min_value=10, max_value=2000, value=30)
sgot = st.number_input("SGOT [5 - 40 U/L]", min_value=10, max_value=2000, value=35)
tp = st.number_input("Total Proteins [6.0 - 8.3 g/dL]", min_value=0.0, max_value=10.0, value=6.5)
alb = st.number_input("Albumin [3.5 - 5.0 g/dL]", min_value=0.0, max_value=10.0, value=3.0)
ag_ratio = st.number_input("A/G Ratio [1.0 - 2.5]", min_value=0.0, max_value=3.0, value=1.0)

gender_encoded = 1 if gender == "Male" else 0

input_data = pd.DataFrame([[age, gender_encoded, tb, db, alkphos, sgpt, sgot, tp, alb, ag_ratio]], columns=X.columns)

# Function to check cirrhosis signs
def check_for_cirrhosis(tb, db, sgpt, sgot, alb, ag_ratio):
    if tb > 2.0 and db > 1.0 and sgpt > 75 and sgot > 75 and alb < 3.0 and ag_ratio < 1.0:
        return True
    return False

if st.button("Predict Liver Condition"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        if check_for_cirrhosis(tb, db, sgpt, sgot, alb, ag_ratio):
            st.warning("⚠️ The model predicts liver disorder — likely **Cirrhosis** based on enzyme levels.")
        else:
            st.warning("⚠️ The model predicts liver disorder — enzyme pattern **suggests other liver issue, not cirrhosis**.")
    else:
        st.success("✅ The model predicts a **healthy liver condition**.")
