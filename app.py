import streamlit as st
from ml_utils import inject_css

st.set_page_config(page_title="Web Model - ML Showcase", page_icon="🤖", layout="wide")
inject_css()

# แต่ละโมเดลเป็นหน้าเว็บแยกกัน มี URL ของตัวเอง (เช่น /knn, /svm)
pages = [
    st.Page("pages/0_Home.py", title="หน้าแรก", icon="🏠", url_path="", default=True),
    st.Page("pages/1_KNN.py", title="KNN", icon="1️⃣", url_path="knn"),
    st.Page("pages/2_Decision_Tree.py", title="Decision Tree", icon="2️⃣", url_path="decision-tree"),
    st.Page("pages/3_SVM.py", title="SVM", icon="3️⃣", url_path="svm"),
    st.Page("pages/4_KMeans.py", title="K-Means", icon="4️⃣", url_path="kmeans"),
    st.Page("pages/5_Regression.py", title="Regression", icon="5️⃣", url_path="regression"),
    st.Page("pages/6_Ensemble_RandomForest.py", title="Random Forest", icon="6️⃣", url_path="random-forest"),
]

# เมนูนำทางอยู่ด้านบนแบบเว็บไซต์ ส่วน sidebar เหลือไว้สำหรับตั้งค่าโมเดลของแต่ละหน้า
nav = st.navigation(pages, position="top")
nav.run()
