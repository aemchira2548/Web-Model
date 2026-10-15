import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC

from ml_utils import load_csv, show_classification_metrics, page_header

page_header("3️⃣", "Support Vector Machine (SVM)", "ทำนายระดับความเครียดของนักเรียน/นักศึกษา (Stress Level)",
            task_type="Classification", color="#EA580C")

df = load_csv("data/student_stress.csv")
TARGET = "Stress_Level"
NUM_FEATURES = ["Sleep_Hours", "Study_Hours", "Social_Media_Hours",
                 "Attendance", "Exam_Pressure", "Family_Support", "Month"]
CAT_FEATURES = ["Student_Type"]
FEATURES = NUM_FEATURES + CAT_FEATURES

with st.expander("📊 ดูข้อมูลชุดข้อมูล (Dataset)"):
    st.write(f"จำนวนแถว: {df.shape[0]} | จำนวนคอลัมน์: {df.shape[1]}")
    st.dataframe(df.head(10), use_container_width=True)
    st.caption("หมายเหตุ: ข้อมูลมีค่าว่าง (missing values) บางส่วน จะถูกเติมด้วยค่ามัธยฐาน/ค่าที่พบบ่อยที่สุดก่อนเทรนโมเดล")

st.sidebar.header("ตั้งค่าโมเดล")
sample_n = st.sidebar.slider("จำนวนตัวอย่างที่ใช้เทรน (สุ่ม)", 500, 5000, 2000, 500)
kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly"], index=0)
C_val = st.sidebar.slider("C (Regularization)", 0.1, 10.0, 1.0, 0.1)
test_size = st.sidebar.slider("สัดส่วน Test set", 0.1, 0.4, 0.2, 0.05)

data = df.dropna(subset=[TARGET])
if len(data) > sample_n:
    data = data.sample(n=sample_n, random_state=42)
X = data[FEATURES]
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), NUM_FEATURES),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), CAT_FEATURES),
])

model = Pipeline([
    ("prep", preprocess),
    ("svm", SVC(kernel=kernel, C=C_val, probability=True)),
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.subheader("ผลการประเมินโมเดล")
show_classification_metrics(y_test, y_pred)

st.divider()
st.subheader("ลองทำนายด้วยข้อมูลของคุณ")

col1, col2, col3 = st.columns(3)
with col1:
    student_type = st.selectbox("ประเภทนักเรียน", ["school", "college"])
    sleep_hours = st.number_input("ชั่วโมงการนอน", 0.0, 14.0, 7.0, 0.1)
    study_hours = st.number_input("ชั่วโมงการเรียน/อ่านหนังสือ", 0.0, 15.0, 3.0, 0.1)
with col2:
    social_media = st.number_input("ชั่วโมงโซเชียลมีเดีย", 0.0, 15.0, 3.0, 0.1)
    attendance = st.number_input("การเข้าเรียน (%)", 0.0, 100.0, 85.0, 0.5)
    exam_pressure = st.slider("ความกดดันเรื่องสอบ (0-10)", 0.0, 10.0, 5.0)
with col3:
    family_support = st.slider("แรงสนับสนุนจากครอบครัว (0-10)", 0.0, 10.0, 5.0)
    month = st.slider("เดือน", 1, 12, 6)

if st.button("ทำนายผล", type="primary"):
    input_df = X.iloc[[0]].copy()
    input_df.loc[:, NUM_FEATURES] = [[sleep_hours, study_hours, social_media,
                                       attendance, exam_pressure, family_support, month]]
    input_df.loc[:, "Student_Type"] = student_type
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][pred]
    if pred == 1:
        st.error(f"⚠️ มีแนวโน้มความเครียดสูง (ความมั่นใจ {proba:.1%})")
    else:
        st.success(f"✅ มีแนวโน้มความเครียดต่ำ (ความมั่นใจ {proba:.1%})")
