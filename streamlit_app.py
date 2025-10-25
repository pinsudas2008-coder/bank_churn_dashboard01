# strekaggle datasets download -d radheshyamkollipara/bank-customer-churn
import streamlit as st
import pandas as pd, numpy as np, os, joblib
import plotly.express as px
import plotly.graph_objects as go
from utils.preprocess import load_data, basic_clean, add_engineered_features, encode_and_scale, split_xy, balance_with_smote
from sklearn.metrics import confusion_matrix, roc_curve, auc

st.set_page_config(layout="wide", page_title="Bank Churn Dashboard ")

DATA_PATH = "data/Churn_Modelling.csv"
KASIKORN_GREEN = "#007A33"

@st.cache_data
def load_prep():
    df = load_data(DATA_PATH)
    df = basic_clean(df)
    df = add_engineered_features(df)
    return df

def overview(df):
    st.title("Bank Customer Churn Dashboard — Kasikorn Theme")
    total = len(df)
    churn = int(df['Exited'].sum())
    active = total - churn
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total customers", total)
    c2.metric("Active customers", active, f"{round(100*active/total,1)}%")
    c3.metric("Churn customers", churn, f"{round(100*churn/total,1)}%")
    c4.metric("Avg credit score", int(df['CreditScore'].mean()))
    st.markdown("---")
    fig = px.pie(names=['Stayed','Exited'], values=[active,churn], title='Customer status (pie)')
    st.plotly_chart(fig, use_container_width=True)
    if 'Geography' in df.columns:
        by_country = df.groupby('Geography')['Exited'].mean().reset_index().sort_values('Exited', ascending=False)
        fig2 = px.bar(by_country, x='Geography', y='Exited', title='Churn rate by country')
        st.plotly_chart(fig2, use_container_width=True)

def eda(df):
    st.header("EDA — grouped by 4 categories (Customer Info, Product Usage, Score, Payment)")
    st.subheader("Customer Information")
    if 'Geography' in df.columns:
        fig = px.histogram(df, x='Geography', color='Exited', barmode='group', title='Country vs Exited')
        st.plotly_chart(fig, use_container_width=True)
    if 'Gender' in df.columns:
        fig = px.bar(df.groupby('Gender')['Exited'].mean().reset_index(), x='Gender', y='Exited', title='Churn rate by Gender')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Product Usage")
    if 'Tenure' in df.columns:
        fig = px.histogram(df, x='Tenure', color='Exited', barmode='overlay', nbins=10, title='Tenure distribution by Exited')
        st.plotly_chart(fig, use_container_width=True)
    if 'NumOfProducts' in df.columns:
        fig = px.box(df, x='NumOfProducts', y='Balance', title='Balance by NumOfProducts')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Customer Score Data")
    if 'CreditScore' in df.columns:
        fig = px.histogram(df, x='CreditScore', color='Exited', barmode='overlay', nbins=30, title='CreditScore distribution by Exited')
        st.plotly_chart(fig, use_container_width=True)
    if 'Satisfaction Score' in df.columns:
        fig = px.bar(df.groupby('Satisfaction Score')['Exited'].mean().reset_index(), x='Satisfaction Score', y='Exited', title='Satisfaction vs Churn rate')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Payment Data")
    if 'Balance' in df.columns:
        fig = px.box(df, y='Balance', title='Balance boxplot')
        st.plotly_chart(fig, use_container_width=True)
    if 'EstimatedSalary' in df.columns:
        fig = px.scatter(df, x='EstimatedSalary', y='Balance', color='Exited', title='Salary vs Balance colored by Exited')
        st.plotly_chart(fig, use_container_width=True)

def preprocessing(df):
    st.header("Preprocessing")
    st.write("Missing values and quick stats")
    miss = df.isnull().sum().to_frame('missing_count')
    miss['missing_pct'] = miss['missing_count']/len(df)*100
    st.dataframe(miss)
    st.write("Feature engineering applied: TenureByAge, BalanceSalaryRatio, CreditScoreGivenAge")
    show_cols = [c for c in ['TenureByAge','BalanceSalaryRatio','CreditScoreGivenAge'] if c in df.columns]
    st.dataframe(df[show_cols].head())

def modeling(df):
    st.header("Modeling & Prediction")
    st.write("Train quick RandomForest here for demo; run model_train.py for full experiments.")
    if st.button("Quick train RF"):
        from sklearn.ensemble import RandomForestClassifier
        df2 = df.copy()
        df2['Exited'] = df2['Exited'].astype(int)
        df_enc, le_map, scaler = encode_and_scale(df2, scaler=None, fit_scaler=True)
        X_train, X_test, y_train, y_test = split_xy(df_enc)
        X_res, y_res = balance_with_smote(X_train, y_train)
        rf = RandomForestClassifier(n_estimators=150, random_state=42)
        rf.fit(X_res, y_res)
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)[:,1]
        acc = (y_pred==y_test).mean()
        st.write("Accuracy:", acc)
        cm = confusion_matrix(y_test, y_pred)
        fig = go.Figure(data=go.Heatmap(z=cm, x=['Pred Stayed','Pred Exited'], y=['True Stayed','True Exited'], colorscale='Greens'))
        fig.update_layout(title='Confusion Matrix')
        st.plotly_chart(fig, use_container_width=True)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC={roc_auc:.3f}'))
        roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash')))
        roc_fig.update_layout(title='ROC Curve')
        st.plotly_chart(roc_fig, use_container_width=True)
        feat_imp = rf.feature_importances_
        cols = X_test.columns.tolist()
        imp_df = pd.DataFrame({'feature':cols, 'importance':feat_imp}).sort_values('importance', ascending=False).head(20)
        fig2 = px.bar(imp_df, x='importance', y='feature', orientation='h', title='Top feature importance')
        st.plotly_chart(fig2, use_container_width=True)
        joblib.dump(rf, "models/rf_model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")
        st.success("Quick model trained and saved to models/rf_model.pkl")

    st.subheader("Predict single customer")
    with st.form("predict_form"):
        cs = st.number_input("CreditScore", value=650, min_value=300, max_value=900)
        age = st.number_input("Age", value=35, min_value=18, max_value=100)
        tenure = st.number_input("Tenure", value=3, min_value=0, max_value=10)
        balance = st.number_input("Balance", value=0.0, step=100.0)
        products = st.number_input("NumOfProducts", value=1, min_value=1, max_value=4)
        hascard = st.selectbox("HasCrCard", [0,1])
        isactive = st.selectbox("IsActiveMember", [0,1])
        sat = st.number_input("Satisfaction Score (1-5)", value=4, min_value=1, max_value=5)
        submitted = st.form_submit_button("Predict")
        if submitted:
            if not os.path.exists("models/rf_model.pkl"):
                st.warning("Model not found. Run quick train or model_train.py first.")
            else:
                mf = joblib.load("models/rf_model.pkl")
                scaler = joblib.load("models/scaler.pkl") if os.path.exists("models/scaler.pkl") else None
                features = np.array([cs, age, tenure, balance, products, hascard, isactive, sat]).reshape(1,-1)
                if scaler is not None:
                    try:
                        features = scaler.transform(features)
                    except Exception:
                        pass
                prob = mf.predict_proba(features)[0,1]
                pred = "Exited" if prob>=0.5 else "Stayed"
                st.metric("Prediction", pred, f"{prob*100:.1f}% chance of churn")

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Overview","EDA","Preprocessing","Model & Predict"])
    if not os.path.exists(DATA_PATH):
        st.error("Place dataset file 'Churn_Modelling.csv' into the data/ folder before using this app.")
        return
    df = load_prep()
    if page=="Overview":
        overview(df)
    elif page=="EDA":
        eda(df)
    elif page=="Preprocessing":
        preprocessing(df)
    elif page=="Model & Predict":
        modeling(df)

if __name__ == '__main__':
    main()