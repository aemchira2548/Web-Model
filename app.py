import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Web Model - ML Showcase", page_icon="🤖", layout="wide")

STUDENT_NAME = "นางสาวเอมจิรา อ่วมเจริญ"
STUDENT_ID = "664245051"
STUDENT_GROUP = "66/44"

PHOTO_PATH = Path(__file__).parent / "assets" / "profile.jpg"

st.title("🤖 Web Model : Machine Learning Showcase")
st.caption("รวม 6 เทคนิค Machine Learning ในเว็บไซต์เดียว (Streamlit)")

col1, col2 = st.columns([1, 3])

with col1:
    if PHOTO_PATH.exists():
        st.image(str(PHOTO_PATH), width=220)
    else:
        st.markdown(
            """
            <div style="width:220px;height:220px;border-radius:12px;
                        background:#e9ecef;display:flex;align-items:center;
                        justify-content:center;font-size:64px;color:#adb5bd;
                        border:2px dashed #adb5bd;">
                📷
            </div>
            <p style="color:#888;font-size:12px;margin-top:6px;">
                วางไฟล์รูปที่ assets/profile.jpg เพื่อแสดงรูปจริง
            </p>
            """,
            unsafe_allow_html=True,
        )

with col2:
    st.subheader("ข้อมูลผู้พัฒนา")
    st.markdown(
        f"""
        - **ชื่อ-นามสกุล:** {STUDENT_NAME}
        - **รหัสนักศึกษา:** {STUDENT_ID}
        - **หมู่เรียน:** {STUDENT_GROUP}
        """
    )

st.divider()

st.subheader("โมเดลทั้งหมดในเว็บไซต์นี้")
st.markdown("เลือกหน้าโมเดลจากเมนูด้านซ้าย (sidebar) เพื่อดูรายละเอียดแต่ละเทคนิค")

models = [
    ("1️⃣ K-Nearest Neighbor", "ทำนายโรคหัวใจ (Heart Disease) จากข้อมูลผู้ป่วย", "pages/1_KNN.py"),
    ("2️⃣ Decision Tree", "ทำนายโรคหัวใจ (Heart Disease) ด้วยต้นไม้ตัดสินใจ", "pages/2_Decision_Tree.py"),
    ("3️⃣ SVM", "ทำนายระดับความเครียดของนักเรียน/นักศึกษา", "pages/3_SVM.py"),
    ("4️⃣ K-Means", "จัดกลุ่มผู้ขอสินเชื่อ (Loan Applicants) แบบ Unsupervised", "pages/4_KMeans.py"),
    ("5️⃣ Regression", "ทำนายคะแนนการบ้าน (Assignment Score) ของนักเรียน", "pages/5_Regression.py"),
    ("6️⃣ Ensemble (Random Forest)", "ทำนายผลการสอบผ่าน/ไม่ผ่านของนักเรียน", "pages/6_Ensemble_RandomForest.py"),
]

for title, desc, _ in models:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(desc)

st.divider()
st.caption("Web Model Assignment — Streamlit Application")
