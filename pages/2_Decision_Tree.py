import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ml_utils import load_csv, show_classification_metrics, page_header

st.set_page_config(page_title="KNN - Heart Disease", page_icon="1️⃣", layout="wide")
page_header("2️⃣", "Decision Tree", "ทำนายโรคหัวใจ (Heart Disease) ด้วยต้นไม้ตัดสินใจ",
            task_type="Classification", color="#0D9488")

df = load_csv("data/heart.csv")
FEATURES = [c for c in df.columns if c != "HeartDisease"]
TARGET = "HeartDisease"

with st.expander("📊 ดูข้อมูลชุดข้อมูล (Dataset)"):
    st.write(f"จำนวนแถว: {df.shape[0]} | จำนวนคอลัมน์: {df.shape[1]}")
    st.dataframe(df.head(10), use_container_width=True)

st.sidebar.header("ตั้งค่าโมเดล")
max_depth = st.sidebar.slider("ความลึกสูงสุด (max_depth)", 1, 15, 4)
test_size = st.sidebar.slider("สัดส่วน Test set", 0.1, 0.4, 0.2, 0.05)

X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.subheader("ผลการประเมินโมเดล")
show_classification_metrics(y_test, y_pred)

st.subheader("ความสำคัญของฟีเจอร์ (Feature Importance)")
importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)
st.bar_chart(importance)

st.divider()
st.subheader("ลองทำนายด้วยข้อมูลของคุณ")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex (1=ชาย, 0=หญิง)", [1, 0])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    resting_bp = st.number_input("Resting Blood Pressure", 60, 220, 130)
with col2:
    chol = st.number_input("Cholesterol", 0, 700, 240)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 (1=ใช่, 0=ไม่ใช่)", [0, 1])
    resting_ecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
    max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
with col3:
    ex_angina = st.selectbox("Exercise Angina (1=ใช่, 0=ไม่ใช่)", [0, 1])
    oldpeak = st.number_input("Oldpeak (ST Depression)", -3.0, 7.0, 1.0, 0.1)
    st_slope = st.selectbox("ST Slope (0-2)", [0, 1, 2])

if st.button("ทำนายผล", type="primary"):
    input_df = X.iloc[[0]].copy()
    input_df.loc[:, :] = [[age, sex, cp, resting_bp, chol, fasting_bs,
                            resting_ecg, max_hr, ex_angina, oldpeak, st_slope]]
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][pred]
    if pred == 1:
        st.error(f"⚠️ มีความเสี่ยงเป็นโรคหัวใจ (ความมั่นใจ {proba:.1%})")
    else:
        st.success(f"✅ ไม่มีความเสี่ยงเป็นโรคหัวใจ (ความมั่นใจ {proba:.1%})")
