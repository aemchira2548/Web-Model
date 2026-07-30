import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from ml_utils import load_csv, show_classification_metrics

st.set_page_config(page_title="Ensemble - Student Pass Prediction", page_icon="6️⃣", layout="wide")
st.title("6️⃣ Ensemble (Random Forest)")
st.caption("ทำนายผลการสอบผ่าน/ไม่ผ่านของนักเรียน (Pass Prediction)")

df = load_csv("data/student_performance.csv")
TARGET = "passed"
NUM_FEATURES = ["study_hours", "attendance_percent", "assignment_score", "previous_gpa"]
CAT_FEATURES = ["internet_access", "tutoring"]
FEATURES = NUM_FEATURES + CAT_FEATURES

with st.expander("📊 ดูข้อมูลชุดข้อมูล (Dataset)"):
    st.write(f"จำนวนแถว: {df.shape[0]} | จำนวนคอลัมน์: {df.shape[1]}")
    st.dataframe(df.head(10), use_container_width=True)
    st.caption("หมายเหตุ: ข้อมูลมีค่าว่าง (missing values) บางส่วน จะถูกเติมด้วยค่ามัธยฐาน/ค่าที่พบบ่อยที่สุดก่อนเทรนโมเดล")

st.sidebar.header("ตั้งค่าโมเดล")
n_estimators = st.sidebar.slider("จำนวนต้นไม้ (n_estimators)", 10, 300, 100, 10)
max_depth = st.sidebar.slider("ความลึกสูงสุด (max_depth)", 1, 20, 6)
test_size = st.sidebar.slider("สัดส่วน Test set", 0.1, 0.4, 0.2, 0.05)

data = df.dropna(subset=[TARGET])
X = data[FEATURES]
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), NUM_FEATURES),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), CAT_FEATURES),
])

model = Pipeline([
    ("prep", preprocess),
    ("rf", RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)),
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.subheader("ผลการประเมินโมเดล")
show_classification_metrics(y_test, y_pred)

st.subheader("ความสำคัญของฟีเจอร์ (Feature Importance)")
feature_names = model.named_steps["prep"].get_feature_names_out()
importance = pd.Series(
    model.named_steps["rf"].feature_importances_, index=feature_names
).sort_values(ascending=True)
st.bar_chart(importance)

st.divider()
st.subheader("ลองทำนายด้วยข้อมูลของคุณ")

col1, col2 = st.columns(2)
with col1:
    study_hours = st.number_input("ชั่วโมงการเรียนต่อวัน", 0.0, 15.0, 4.0, 0.1)
    attendance_percent = st.number_input("การเข้าเรียน (%)", 0.0, 100.0, 85.0, 0.5)
    assignment_score = st.number_input("คะแนนการบ้าน", 0.0, 100.0, 70.0, 0.5)
    previous_gpa = st.number_input("GPA เดิม", 0.0, 4.0, 3.0, 0.01)
with col2:
    internet_access = st.selectbox("เข้าถึงอินเทอร์เน็ต", ["Yes", "No"])
    tutoring = st.selectbox("มีติวเสริม (Tutoring)", ["Yes", "No"])

if st.button("ทำนายผล", type="primary"):
    input_df = X.iloc[[0]].copy()
    input_df.loc[:, NUM_FEATURES] = [[study_hours, attendance_percent, assignment_score, previous_gpa]]
    input_df.loc[:, "internet_access"] = internet_access
    input_df.loc[:, "tutoring"] = tutoring
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][pred]
    if pred == 1:
        st.success(f"✅ คาดการณ์ว่า **สอบผ่าน** (ความมั่นใจ {proba:.1%})")
    else:
        st.error(f"⚠️ คาดการณ์ว่า **สอบไม่ผ่าน** (ความมั่นใจ {proba:.1%})")
