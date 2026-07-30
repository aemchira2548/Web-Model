# Web Model : Machine Learning Showcase

เว็บแอปพลิเคชัน Streamlit ที่รวม 6 เทคนิค Machine Learning ไว้ในที่เดียว

**ผู้พัฒนา:** นางสาวเอมจิรา อ่วมเจริญ | รหัสนักศึกษา 664245051 | หมู่เรียน 66/44

## โมเดลที่ใช้

| # | โมเดล | Dataset | งาน |
|---|-------|---------|-----|
| 1 | K-Nearest Neighbor | Heart Disease | Classification |
| 2 | Decision Tree | Heart Disease | Classification |
| 3 | SVM | Student Lifestyle & Stress | Classification |
| 4 | K-Means | Loan Applicants | Clustering |
| 5 | Regression (Linear) | Student Performance | Regression |
| 6 | Ensemble (Random Forest) | Student Performance | Classification |

## วิธีรันโปรเจกต์

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

จากนั้นเปิดเบราว์เซอร์ที่ `http://localhost:8501`

## โครงสร้างโปรเจกต์

```
web-model-ml/
├── app.py                          # หน้าแรก + ข้อมูลผู้พัฒนา
├── ml_utils.py                     # ฟังก์ชันช่วยเหลือ (โหลดข้อมูล, แสดงผล metrics)
├── pages/
│   ├── 1_KNN.py
│   ├── 2_Decision_Tree.py
│   ├── 3_SVM.py
│   ├── 4_KMeans.py
│   ├── 5_Regression.py
│   └── 6_Ensemble_RandomForest.py
├── data/                           # ชุดข้อมูลทั้งหมด (CSV)
├── assets/                         # วางไฟล์ profile.jpg ที่นี่เพื่อแสดงรูปโปรไฟล์
└── requirements.txt
```

## หมายเหตุ

- วางรูปโปรไฟล์ที่ `assets/profile.jpg` เพื่อให้แสดงบนหน้าแรกแทนกล่องว่าง
- ข้อมูลบางชุด (Student Stress, Student Performance) มีค่าว่างบางส่วน ซึ่งแต่ละหน้าจะเติมค่าด้วย median/most-frequent ก่อนเทรนโมเดลโดยอัตโนมัติ
