import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# ------------------------------
st.set_page_config(page_title="AI Cyber Fraud Detection", page_icon="💳", layout="wide", initial_sidebar_state="expanded")

# --- [ CYBERPUNK CSS - KEEP IT ] ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Roboto:wght@300;400;700&display=swap');
    .stApp { background: radial-gradient(circle at top left, #1A1A2E 0%, #0F0F1A 40%, #000000 100%); color: #E0E6ED; font-family: 'Roboto', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Orbitron', sans-serif !important; color: #00F0FF !important; text-shadow: 0 0 10px rgba(0, 240, 255, 0.6); }
    div[data-testid="stMetric"] { background: rgba(20, 30, 50, 0.6); backdrop-filter: blur(12px); border-radius: 12px; padding: 20px; border: 1px solid rgba(0, 240, 255, 0.2); }
    div[data-testid="stMetric"]:hover { transform: translateY(-5px); border-color: #00F0FF; box-shadow: 0 0 25px rgba(0, 240, 255, 0.4); }
    .stButton > button { background: linear-gradient(45deg, #00F0FF 0%, #0077FF 100%); color: #050A14; border: none; border-radius: 8px; padding: 12px 28px; font-weight: 800; font-family: 'Orbitron', sans-serif; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0D1117 0%, #161B22 100%); border-right: 1px solid rgba(0, 240, 255, 0.3); }
</style>
""", unsafe_allow_html=True)

# ------------------------------
@st.cache_resource
def load_model(): return joblib.load("model.pkl")
@st.cache_data
def load_data(): return pd.read_csv("creditcard.csv.csv")
@st.cache_data
def compute_performance(_data, _model):
    X = _data.drop("Class", axis=1); y = _data["Class"]
    y_pred = _model.predict(X); y_proba = _model.predict_proba(X)[:, 1]
    return y, y_pred, y_proba

model = load_model()
data = load_data()  # Always load default dataset

# Compute performance (Class column exists)
true_labels, predicted_labels, predicted_proba = compute_performance(data, model)
acc = accuracy_score(true_labels, predicted_labels)
prec = precision_score(true_labels, predicted_labels)
rec = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

# Basic stats
fraud_count = (data["Class"] == 1).sum()
normal_count = len(data) - fraud_count
total_count = len(data)
fraud_ratio = fraud_count / total_count * 100

# Sidebar
with st.sidebar:
    st.markdown("<h1 style='text-align:center;'>CYBER DASHBOARD</h1>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063439.png", width=80)
    st.markdown("---")
    st.metric("Total Records", f"{total_count:,}")
    st.metric("Frauds", f"{fraud_count:,}")
    st.metric("Fraud Ratio", f"{fraud_ratio:.3f}%")
    st.markdown("---")
    show_eda = st.checkbox("📈 EDA", True)
    show_model = st.checkbox("🤖 Model Performance", True)
    show_features = st.checkbox("🌟 Feature Importance", True)
    
    # Download Predictions (always available because we have Class)
    results_df = data.copy()
    results_df['Predicted'] = predicted_labels
    results_df['Fraud_Probability'] = predicted_proba
    csv = results_df.to_csv(index=False)
    st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")

# ------------------------------
st.markdown("<h1 style='text-align: center;'>💳 AI-POWERED FRAUD DETECTION</h1>", unsafe_allow_html=True)

# Project Description
st.info("This dashboard uses a **Random Forest Machine Learning model** to identify fraudulent credit card transactions in real time. Upload your data or test the model live.")

# Hero Accuracy
st.success(f"🎯 Model Accuracy: {acc:.2%} | Trained on 284,807 Transactions")

# Section 1: Overview
col1, col2 = st.columns([2, 3])
with col1:
    fig_pie = px.pie(names=["Normal", "Fraud"], values=[normal_count, fraud_count], color_discrete_sequence=["#00F0FF", "#FF007F"], hole=0.5)
    fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=300, margin=dict(t=0,b=0))
    st.plotly_chart(fig_pie, use_container_width=True)

# Section 2: EDA (if checked)
if show_eda:
    st.markdown("---")
    st.header("📈 Exploratory Data Analysis")
    tab1, tab2, tab3 = st.tabs(["💰 Amount & Time", "🧊 3D Clusters", "🔥 Correlations"])
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig_hist = px.histogram(data.sample(min(10000, len(data))), x="Amount", color="Class", nbins=80,
                                    color_discrete_map={0: "#00F0FF", 1: "#FF007F"}, title="Transaction Amount Distribution",
                                    opacity=0.7, barmode='overlay')
            fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig_hist, use_container_width=True)
        with c2:
            data['Hour'] = data['Time'] / 3600
            fig_box = px.box(data.sample(min(10000, len(data))), x="Class", y="Amount", color="Class",
                             color_discrete_map={0: "#00F0FF", 1: "#FF007F"}, title="Amount by Class")
            fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig_box, use_container_width=True)
    with tab2:
        sample_3d = data.sample(min(2000, len(data)))
        fig_3d = px.scatter_3d(sample_3d, x='Time', y='Amount', z='V1', color='Class',
                               color_discrete_map={0: '#00F0FF', 1: '#FF007F'},
                               title="3D Transaction Clusters (Time vs Amount vs V1)",
                               opacity=0.7, size_max=10)
        fig_3d.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                             font=dict(color='white'), scene=dict(
                                 xaxis=dict(backgroundcolor='rgba(0,0,0,0)', color='white', title='Time (s)'),
                                 yaxis=dict(backgroundcolor='rgba(0,0,0,0)', color='white', title='Amount ($)'),
                                 zaxis=dict(backgroundcolor='rgba(0,0,0,0)', color='white', title='V1 (PCA)')
                             ))
        st.plotly_chart(fig_3d, use_container_width=True)
    with tab3:
        corr = data.drop(columns=['Hour'], errors='ignore').corr()
        top_corr = corr['Class'].abs().sort_values(ascending=False).head(15).index
        corr_matrix = data[top_corr].corr()
        fig_heat = px.imshow(corr_matrix, color_continuous_scale='Inferno', title="Top Feature Correlation Matrix")
        fig_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig_heat, use_container_width=True)

# Section 3: Model Performance
if show_model:
    st.markdown("---")
    st.header("🤖 Model Performance")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("Precision", f"{prec:.4f}")
    c3.metric("Recall", f"{rec:.4f}")
    c4.metric("F1", f"{f1:.4f}")

    fpr, tpr, _ = roc_curve(true_labels, predicted_proba)
    roc_auc = auc(fpr, tpr)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, line=dict(color='#00F0FF', width=3), name=f'AUC = {roc_auc:.4f}', fill='tozeroy', fillcolor='rgba(0, 240, 255, 0.1)'))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash='dash', color='#FF007F'), name='Random Guess'))
    fig_roc.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    st.plotly_chart(fig_roc, use_container_width=True)

# Section 4: Feature Importance
if show_features and hasattr(model, "feature_importances_"):
    st.markdown("---")
    st.header("🌟 Feature Importance")
    feature_names = [c for c in data.columns if c not in ['Class', 'Hour']]
    importances = model.feature_importances_
    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=True).tail(15)
    fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Turbo', title="Top 15 Features")
    fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    st.plotly_chart(fig_imp, use_container_width=True)

# Section 5: Prediction Lab
st.markdown("---")
st.header("🔮 Prediction Lab")
idx = st.number_input("Select Transaction Index", 0, len(data)-1, 0)
sample = data.drop(columns=['Class'], errors='ignore').iloc[[idx]]
pred = model.predict(sample)[0]
proba = model.predict_proba(sample)[0][1]

col1, col2 = st.columns(2)
with col1:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=proba*100,
        title={'text': "Fraud Probability (%)", 'font': {'color': 'white'}},
        number={'font': {'color': 'white', 'size': 40}},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#FF007F" if proba > 0.5 else "#00F0FF"},
               'steps': [{'range': [0, 50], 'color': 'rgba(0, 240, 255, 0.2)'}, {'range': [50, 100], 'color': 'rgba(255, 0, 127, 0.2)'}],
               'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': 50}}
    ))
    fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=250)
    st.plotly_chart(fig_gauge, use_container_width=True)
with col2:
    if pred == 0:
        st.success(f"✅ **Safe Transaction**\n\nProbability of Fraud: {proba:.2%}")
    else:
        st.error(f"🚨 **Fraudulent Transaction**\n\nProbability of Fraud: {proba:.2%}")
    # Download single result
    result_df = sample.copy()
    result_df['Prediction'] = pred
    result_df['Fraud_Probability'] = proba
    st.download_button("📥 Download This Result", result_df.to_csv(index=False), f"result_{idx}.csv")

# Footer with GitHub link
st.markdown("---")
st.markdown("""
<div style='text-align:center'>
<h3>👨‍💻 Developed by <b>Vivian Lobo</b></h3>
<p>AI & ML Internship Project | Random Forest Classifier</p>
<a href='https://github.com/vivianlobo07' target='_blank' style='color:#00F0FF; text-decoration:none; font-size:18px;'>🔗 GitHub: github.com/yourusername</a>
</div>
""", unsafe_allow_html=True)