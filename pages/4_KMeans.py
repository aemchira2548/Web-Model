import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from ml_utils import load_csv

st.set_page_config(page_title="K-Means - Loan Applicants", page_icon="4️⃣", layout="wide")
st.title("4️⃣ K-Means Clustering")
st.caption("จัดกลุ่มผู้ขอสินเชื่อ (Loan Applicants) แบบ Unsupervised Learning")

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
ax2.set_title("การกระจายตัวของกลุ่ม (PCA 2D)")
st.pyplot(fig2, use_container_width=False)

st.subheader("ค่าเฉลี่ยของแต่ละกลุ่ม (Cluster Profile)")
st.dataframe(data.groupby("Cluster")[NUM_FEATURES].mean().round(2), use_container_width=True)

st.subheader("จำนวนสมาชิกในแต่ละกลุ่ม")
st.bar_chart(data["Cluster"].value_counts().sort_index())
