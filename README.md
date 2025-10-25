# bank_churn_dashboard01 
Bank Churn Dashboard 
Dashboard นี้สร้างด้วย Streamlit สำหรับวิเคราะห์และทำนาย Customer Churn ของธนาคาร

ทำ EDA Visualization Preprocessing & Feature Engineering Quick RandomForest Interactive metrics & highlights

# Features Overview:
Total customers, churn rate, average credit score EDA: Visualization ของลูกค้าแยกตาม Geography, Gender, Tenure, Balance, etc. Preprocessing: Missing value check, feature engineering Model & Predict: Quick train RandomForest, predict single customer, feature importance, ROC curve 

1.git clone https://github.com/pinsudas2008-coder/bank_churn_dashboard.git cd bank_churn_dashboard01

2.สร้าง virtual environment 
ใช้คำสั่ง  python -m venv venv
# สำหรับWindows
.venv\Scripts\activate

# สำหรับMac/Linux
source .venv/bin/activate

3.เมื่อได้ venv แล้ว ติดตั้งไลบารี่สำหรับการใช้งาน
ใช้คำสั่ง pip install -r requirements.txt 

4. จากนั้นใช้ คำสั่งรัน python api.py ต่อด้วย python model_train.py
   เมื่อจะใช้ streamlit ต้องใช้คำสั่ง streamlit run streamlit_app.py ก็จะมีหน้าเว็บเด้งขึ้นมา
