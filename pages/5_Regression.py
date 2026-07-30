import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from ml_utils import load_csv

st.set_page_config(page_title="Regression - Student Performance", page_icon="5️⃣", layout="wide")
st.title("5️⃣ Regression")
st.caption("ทำนายคะแนนการบ้าน (Assignment Score) ของนักเรียน")

df = load_csv("data/student_performance.csv")
TARGET = "assignment_score"
NUM_FEATURES = ["study_hours", "attendance_percent", "previous_gpa"]
CAT_FEATURES = ["internet_access", "tutoring"]
FEATURES = NUM_FEATURES + CAT_FEATURES

with st.expander("📊 ดูข้อมูลชุดข้อมูล (Dataset)"):
    st.write(f"จำนวนแถว: {df.shape[0]} | จำนวนคอลัมน์: {df.shape[1]}")
    st.dataframe(df.head(10), use_container_width=True)
    st.caption("หมายเหตุ: ข้อมูลมีค่าว่าง (missing values) บางส่วน จะถูกเติมด้วยค่ามัธยฐาน/ค่าที่พบบ่อยที่สุดก่อนเทรนโมเดล")

st.sidebar.header("ตั้งค่าโมเดล")
test_size = st.sidebar.slider("สัดส่วน Test set", 0.1, 0.4, 0.2, 0.05)

data = df.dropna(subset=[TARGET])
X = data[FEATURES]
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
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
    ("linreg", LinearRegression()),
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

st.subheader("ผลการประเมินโมเดล")
c1, c2, c3 = st.columns(3)
c1.metric("R² Score", f"{r2:.3f}")
c2.metric("MAE", f"{mae:.2f}")
c3.metric("RMSE", f"{rmse:.2f}")

if r2 < 0.1:
    st.info(
        "💡 **ข้อสังเกต:** ค่า R² ต่ำ เนื่องจากตัวแปรต้น (study_hours, attendance_percent, "
        "previous_gpa ฯลฯ) มีค่าสหสัมพันธ์ (correlation) กับ assignment_score ใกล้ศูนย์ในชุดข้อมูลนี้ "
        "จึงเป็นข้อจำกัดของข้อมูล ไม่ใช่ข้อผิดพลาดของโมเดล — สะท้อนว่าโมเดลเชิงเส้นไม่สามารถอธิบาย "
        "ความสัมพันธ์ที่ไม่มีอยู่จริงในข้อมูลได้"
    )

fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(y_test, y_pred, alpha=0.5)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, "r--", label="Perfect Prediction")
ax.set_xlabel("Actual Assignment Score")
ax.set_ylabel("Predicted Assignment Score")
ax.legend()
st.pyplot(fig, use_container_width=False)

st.divider()
st.subheader("ลองทำนายด้วยข้อมูลของคุณ")

col1, col2 = st.columns(2)
with col1:
    study_hours = st.number_input("ชั่วโมงการเรียนต่อวัน", 0.0, 15.0, 4.0, 0.1)
    attendance_percent = st.number_input("การเข้าเรียน (%)", 0.0, 100.0, 85.0, 0.5)
    previous_gpa = st.number_input("GPA เดิม", 0.0, 4.0, 3.0, 0.01)
with col2:
    internet_access = st.selectbox("เข้าถึงอินเทอร์เน็ต", ["Yes", "No"])
    tutoring = st.selectbox("มีติวเสริม (Tutoring)", ["Yes", "No"])

if st.button("ทำนายผล", type="primary"):
    input_df = X.iloc[[0]].copy()
    input_df.loc[:, NUM_FEATURES] = [[study_hours, attendance_percent, previous_gpa]]
    input_df.loc[:, "internet_access"] = internet_access
    input_df.loc[:, "tutoring"] = tutoring
    pred = model.predict(input_df)[0]
    st.info(f"📈 คาดการณ์คะแนนการบ้าน: **{pred:.1f}** คะแนน")
