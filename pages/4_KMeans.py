import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from ml_utils import load_csv, page_header

st.set_page_config(page_title="KNN - Heart Disease", page_icon="1️⃣", layout="wide")
page_header("4️⃣", "K-Means Clustering", "จัดกลุ่มผู้ขอสินเชื่อ (Loan Applicants) แบบ Unsupervised Learning",
            task_type="Clustering", color="#DB2777")

df = load_csv("data/loan_data.csv")
NUM_FEATURES = ["person_age", "person_income", "person_emp_exp", "loan_amnt",
                 "loan_int_rate", "loan_percent_income",
                 "cb_person_cred_hist_length", "credit_score"]

with st.expander("📊 ดูข้อมูลชุดข้อมูล (Dataset)"):
    st.write(f"จำนวนแถว: {df.shape[0]} | จำนวนคอลัมน์: {df.shape[1]}")
    st.dataframe(df.head(10), use_container_width=True)

st.sidebar.header("ตั้งค่าโมเดล")
sample_n = st.sidebar.slider("จำนวนตัวอย่างที่ใช้ (สุ่ม)", 500, 5000, 2000, 500)
k = st.sidebar.slider("จำนวนกลุ่ม (k)", 2, 8, 4)

data = df[NUM_FEATURES].dropna().sample(n=sample_n, random_state=42).reset_index(drop=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

st.subheader("Elbow Method: เลือกจำนวนกลุ่มที่เหมาะสม")
inertias = []
k_range = range(1, 9)
for i in k_range:
    km = KMeans(n_clusters=i, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(list(k_range), inertias, marker="o")
ax.set_xlabel("k")
ax.set_ylabel("Inertia")
ax.set_title("Elbow Curve")
st.pyplot(fig, use_container_width=False)

model = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = model.fit_predict(X_scaled)
data["Cluster"] = clusters

st.subheader(f"ผลการจัดกลุ่ม (k={k})")

pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)
plot_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
plot_df["Cluster"] = clusters.astype(str)

fig2, ax2 = plt.subplots(figsize=(6, 4))
for c in sorted(plot_df["Cluster"].unique()):
    subset = plot_df[plot_df["Cluster"] == c]
    ax2.scatter(subset["PC1"], subset["PC2"], label=f"Cluster {c}", alpha=0.6, s=15)
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax2.legend()
ax2.set_title("Cluster Distribution (PCA 2D)")
st.pyplot(fig2, use_container_width=False)

st.subheader("ค่าเฉลี่ยของแต่ละกลุ่ม (Cluster Profile)")
st.dataframe(data.groupby("Cluster")[NUM_FEATURES].mean().round(2), use_container_width=True)

st.subheader("จำนวนสมาชิกในแต่ละกลุ่ม")
st.bar_chart(data["Cluster"].value_counts().sort_index())

st.divider()
st.subheader("ลองคำนวณด้วยข้อมูลของคุณ")
st.caption("กรอกข้อมูลผู้ขอสินเชื่อ ระบบจะคำนวณว่าใกล้เคียงกับกลุ่มไหนมากที่สุด")

col1, col2, col3, col4 = st.columns(4)
with col1:
    in_age = st.number_input("อายุ", 18, 100, 30)
    in_income = st.number_input("รายได้ต่อปี", 0, 500000, 60000, 1000)
with col2:
    in_emp_exp = st.number_input("ประสบการณ์ทำงาน (ปี)", 0, 50, 3)
    in_loan_amnt = st.number_input("จำนวนเงินกู้", 0, 100000, 10000, 500)
with col3:
    in_int_rate = st.number_input("อัตราดอกเบี้ย (%)", 0.0, 30.0, 12.0, 0.1)
    in_percent_income = st.number_input("สัดส่วนเงินกู้ต่อรายได้", 0.0, 1.0, 0.2, 0.01)
with col4:
    in_hist_length = st.number_input("ความยาวประวัติเครดิต (ปี)", 0, 40, 5)
    in_credit_score = st.number_input("Credit Score", 300, 900, 650)

if st.button("คำนวณกลุ่ม", type="primary"):
    new_point = pd.DataFrame([[in_age, in_income, in_emp_exp, in_loan_amnt,
                                in_int_rate, in_percent_income,
                                in_hist_length, in_credit_score]], columns=NUM_FEATURES)
    new_scaled = scaler.transform(new_point)
    predicted_cluster = model.predict(new_scaled)[0]
    st.success(f"🎯 ข้อมูลนี้ใกล้เคียงกับ **Cluster {predicted_cluster}** มากที่สุด")
    st.write("ค่าเฉลี่ยของกลุ่มนี้เทียบกับข้อมูลที่กรอก:")
    profile = data[data["Cluster"] == predicted_cluster][NUM_FEATURES].mean().round(2)
    compare_df = pd.DataFrame({
        "ค่าเฉลี่ยของกลุ่ม": profile,
        "ข้อมูลที่คุณกรอก": new_point.iloc[0],
    })
    st.dataframe(compare_df, use_container_width=True)
