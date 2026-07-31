import base64
import streamlit as st
from pathlib import Path
from ml_utils import inject_css

st.set_page_config(page_title="Web Model - ML Showcase", page_icon="🤖", layout="wide")
inject_css()

STUDENT_NAME = "นางสาวเอมจิรา อ่วมเจริญ"
STUDENT_ID = "664245051"
STUDENT_GROUP = "66/44"

PHOTO_PATH = Path(__file__).parent / "assets" / "profile.jpg"

st.markdown(
    """
    <div style="background: linear-gradient(120deg, #6C5CE7 0%, #A29BFE 100%);
                border-radius: 20px; padding: 40px 44px; margin-bottom: 26px;
                box-shadow: 0 12px 30px rgba(108,92,231,0.25); color: white;">
        <div style="font-size: 13px; letter-spacing: 2px; opacity: 0.85; font-weight: 600;">
            WEB MODEL &nbsp;•&nbsp; MACHINE LEARNING ASSIGNMENT
        </div>
        <div style="font-size: 34px; font-weight: 800; margin: 8px 0 6px 0;">
            🤖 เว็บไซต์ Machine Learning
        </div>
        <div style="font-size: 15px; opacity: 0.92; max-width: 620px;">
            จัดทำขึ้นเพื่อส่งงานวิชา Web Model โดยรวมโมเดล Machine Learning 6 แบบไว้ในเว็บเดียว
            ได้แก่ KNN, Decision Tree, SVM, K-Means, Regression และ Random Forest
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1, 3], gap="large")

with col1:
    if PHOTO_PATH.exists():
        b64 = base64.b64encode(PHOTO_PATH.read_bytes()).decode()
        st.markdown(
            f"""
            <div style="width:200px;height:200px;border-radius:50%;padding:5px;
                        background: linear-gradient(135deg, #6C5CE7, #FD79A8);
                        display:flex;align-items:center;justify-content:center;">
                <img src="data:image/jpeg;base64,{b64}"
                     style="width:100%;height:100%;border-radius:50%;object-fit:cover;
                            border:3px solid white;" />
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="width:200px;height:200px;border-radius:50%;
                        background: linear-gradient(135deg, #EEF2FF, #F5F3FF);
                        display:flex;align-items:center;justify-content:center;
                        font-size:64px;color:#A29BFE;
                        border:3px dashed #C7C0FB;">
                📷
            </div>
            <p style="color:#9CA3AF;font-size:12px;margin-top:10px;">
                วางไฟล์รูปที่ assets/profile.jpg เพื่อแสดงรูปจริง
            </p>
            """,
            unsafe_allow_html=True,
        )

with col2:
    st.markdown(
        f"""
        <div style="background:#FFFFFF;border:1px solid #ECE9FE;border-radius:16px;
                    padding:26px 28px;box-shadow:0 2px 10px rgba(108,92,231,0.06);">
            <div style="font-size:13px;font-weight:700;color:#6C5CE7;letter-spacing:1px;
                        margin-bottom:14px;">ข้อมูลผู้พัฒนา</div>
            <div style="display:grid;grid-template-columns:140px 1fr;row-gap:10px;font-size:15px;">
                <div style="color:#9CA3AF;">ชื่อ-นามสกุล</div><div style="font-weight:600;">{STUDENT_NAME}</div>
                <div style="color:#9CA3AF;">รหัสนักศึกษา</div><div style="font-weight:600;">{STUDENT_ID}</div>
                <div style="color:#9CA3AF;">หมู่เรียน</div><div style="font-weight:600;">{STUDENT_GROUP}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.markdown("#### โมเดลทั้งหมดในเว็บไซต์นี้")
st.caption("คลิกที่การ์ดด้านล่างเพื่อเปิดหน้าโมเดล หรือเลือกจากเมนูด้านซ้าย (sidebar)")

models = [
    ("1️⃣", "K-Nearest Neighbor", "ทำนายโรคหัวใจจากข้อมูลผู้ป่วย (Heart Disease)", "Classification", "#4F46E5", "#EEF2FF", "pages/1_KNN.py"),
    ("2️⃣", "Decision Tree", "ทำนายโรคหัวใจด้วยต้นไม้ตัดสินใจ (Heart Disease)", "Classification", "#0D9488", "#F0FDFA", "pages/2_Decision_Tree.py"),
    ("3️⃣", "SVM", "ทำนายระดับความเครียดของนักเรียน/นักศึกษา", "Classification", "#EA580C", "#FFF7ED", "pages/3_SVM.py"),
    ("4️⃣", "K-Means", "จัดกลุ่มผู้ขอสินเชื่อ (Loan Applicants) แบบ Unsupervised", "Clustering", "#DB2777", "#FDF2F8", "pages/4_KMeans.py"),
    ("5️⃣", "Regression", "ทำนายคะแนนการบ้าน (Assignment Score) ของนักเรียน", "Regression", "#059669", "#ECFDF5", "pages/5_Regression.py"),
    ("6️⃣", "Ensemble (Random Forest)", "ทำนายผลการสอบผ่าน/ไม่ผ่านของนักเรียน", "Classification", "#7C3AED", "#F5F3FF", "pages/6_Ensemble_RandomForest.py"),
]

badge_bg = {
    "Classification": ("#EEF2FF", "#4F46E5"),
    "Regression": ("#ECFDF5", "#059669"),
    "Clustering": ("#FDF2F8", "#DB2777"),
}

cols = st.columns(3, gap="medium")
for i, (num, title, desc, task, color, icon_bg, page_path) in enumerate(models):
    bg, fg = badge_bg[task]
    with cols[i % 3]:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="m-icon" style="background:{icon_bg};color:{color};">{num}</div>
                <div class="m-title">{title}</div>
                <div class="m-desc">{desc}</div>
                <span class="badge-pill" style="background:{bg};color:{fg};">{task}</span>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            st.page_link(page_path, label="เปิดหน้านี้", icon="👉", use_container_width=True)

st.write("")
st.divider()
st.caption("Web Model Assignment — Streamlit Application")
