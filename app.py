import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ===================== Page Config =====================
st.set_page_config(page_title="Recruitment Prediction App", layout="wide")
ACCENT_COLOR = "#1F3A93"

# ===================== Load Font Awesome =====================
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
""", unsafe_allow_html=True)

# ===================== CSS Custom =====================
st.markdown("""
<style>
.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 25px;
    background: linear-gradient(to right, #2b1e77, #6a0dad);
    color: white;
    font-weight: bold;
    font-size: 30px;
    border-radius: 12px;
}
.topbar .icon { font-size: 30px; color: white; }
.hero-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: white;
    background: linear-gradient(to right, #2b1e77, #6a0dad);
    padding: 20px;
    border-radius: 18px;
    margin: 15px auto;
    width: 85%;
}
.hero-title i { margin-right: 10px; }
.instruction-box {
    background: #D8E6F4;
    color: #000;
    padding: 15px;
    border-radius: 12px;
    font-size: 16px;
    margin: 10px 0;
}
.stButton>button {
    background: linear-gradient(to right, #6a0dad, #2b1e77);
    color: white;
    font-weight: 900;
    border-radius: 25px;
    padding: 10px 24px;
    border: none;
    font-size: 20px;
    cursor: pointer;
    width: 100%;
}
.stButton>button:hover { opacity: 0.9; }
.tbl-header { font-weight:bold; text-align:center; border:1px solid #ccc; padding:8px; background:#E6E6FA;}
.tbl-cell { border:1px solid #ccc; padding:6px; text-align:center; font-size:16px; }
.status-accepted { background-color: #D8F4D7; color:green; font-weight:bold; }
.status-rejected { background-color: #FADBD8; color:red; font-weight:bold; }
[data-baseweb="radio"] label {
    color: white !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)

# ===================== Topbar & Hero =====================
st.markdown("""
<div class="topbar">
    <div>PANCARONA</div>
    <div class="icon"><i class="fas fa-briefcase"></i></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-title">
    <i class="fas fa-user"></i> Recruitment Prediction App
</div>
""", unsafe_allow_html=True)

# ===================== Sidebar =====================
st.sidebar.markdown("""
<div style="
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    text-align:center;
    padding: 20px;
    border-radius: 12px;
    background: linear-gradient(to bottom, #2b1e77, #6a0dad);
    color: white;
">
    <h3 style='font-weight:bold; margin-bottom:15px;'>Recruitment Prediction</h3>
    <div style='background: #FFFFFF;
                color: #000;
                padding: 15px;
                border-radius: 12px;
                font-size: 16px;
                margin-top: 10px;
                width: 90%;
                text-align:center;'>
        Aplikasi ini membantu proses rekrutmen agar lebih efisien dan objektif.
    </div>
    <div style='background: #FFFFFF;
                color: #000;
                padding: 15px;
                border-radius: 12px;
                font-size: 16px;
                margin-top: 10px;
                width: 90%;
                text-align:center;'>
        Fokus utamanya adalah mendukung tahap akhir seleksi dengan memanfaatkan data penilaian kandidat untuk keputusan penerimaan yang lebih tepat.
    </div>
</div>

<div style="margin-top: 20px;">
    <p style='color:black; font-weight:bold; margin-bottom:5px;'>Pilih Halaman</p>
</div>
""", unsafe_allow_html=True)

# Radio button halaman
page_options = ["Input Data", "Prediksi", "EDA"]
page = st.sidebar.radio("", page_options, label_visibility="collapsed")

# ===================== Load Model & Scaler =====================
model, scaler = None, None
if os.path.exists("best_catboost_optuna.joblib") and os.path.exists("scaler.joblib"):
    model = joblib.load("best_catboost_optuna.joblib")
    scaler = joblib.load("scaler.joblib")
else:
    st.warning("⚠ File model atau scaler tidak ditemukan. Prediksi tidak bisa dijalankan.")

# ===================== Preprocess =====================
def preprocess_input(df, scaler):
    numeric_cols = ['ExperienceYears','InterviewScore','SkillScore','PersonalityScore']
    df['TotalScore'] = df['SkillScore'] + df['InterviewScore'] + df['PersonalityScore']
    df['Skill_Experience_Interaction'] = df['SkillScore'] * df['ExperienceYears']

    # One-hot recruitment
    rec_map = {1:[1,0,0], 2:[0,1,0], 3:[0,0,1]}
    rec_cols = ['RecruitmentStrategy_1','RecruitmentStrategy_2','RecruitmentStrategy_3']
    df[rec_cols] = df['RecruitmentStrategy'].map(lambda x: rec_map.get(x,[0,0,0])).apply(pd.Series)

    # Normalisasi numeric tetap di balik layar
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    selected_features = ['EducationLevel','ExperienceYears','InterviewScore','SkillScore','PersonalityScore',
                         'TotalScore','RecruitmentStrategy_1','RecruitmentStrategy_2','RecruitmentStrategy_3',
                         'Skill_Experience_Interaction']
    return df[selected_features]

# ===================== Halaman Input Data =====================
if page == "Input Data":
    st.markdown("<h2 style='text-align:center; background:linear-gradient(to right,#2b1e77,#6a0dad); color:white; padding:12px; border-radius:12px;'><i class='fas fa-file-upload'></i> Input Data Kandidat</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div class="instruction-box">
    Threshold diterima  : 61% <br>
    Model               : Catboost Optuna  <br>
    F1-Score            : 0.913
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="instruction-box">
    File CSV harus berisi kolom:<br>
    - Name<br>
    - EducationLevel<br>
    - ExperienceYears<br>
    - InterviewScore<br>
    - SkillScore<br>
    - PersonalityScore<br>
    - RecruitmentStrategy
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload File CSV", type=['csv'])
    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)
        st.session_state.df_input = df_input
        st.info(f"Total data: {len(df_input)} baris")
        st.dataframe(df_input.head(10))

        if st.button("Prediksi"):
            if model and scaler:
                df_pred = df_input.copy()
                processed = preprocess_input(df_pred, scaler)
                processed = processed[['EducationLevel','ExperienceYears','InterviewScore','SkillScore','PersonalityScore',
                                       'TotalScore','RecruitmentStrategy_1','RecruitmentStrategy_2','RecruitmentStrategy_3',
                                       'Skill_Experience_Interaction']]
                prob = model.predict_proba(processed)[:,1]*100
                df_pred['Probability'] = prob
                df_pred['Status'] = np.where(prob>=61,"Diterima","Tidak Diterima")
                st.session_state.pred_df = df_pred
                st.success("Prediksi selesai. Lihat halaman 'Prediksi'.")

# ===================== Halaman Prediksi =====================
elif page == "Prediksi":
    st.markdown("<h2 style='text-align:center; background:linear-gradient(to right,#2b1e77,#6a0dad); color:white; padding:12px; border-radius:12px; margin-bottom:15px;'><i class='fas fa-chart-line'></i> Hasil Prediksi</h2>", unsafe_allow_html=True)

    if "pred_df" in st.session_state:
        df_pred = st.session_state.pred_df
        preview_df = df_pred.head(15).copy()
        preview_df_display = preview_df.drop(columns=['RecruitmentStrategy_1','RecruitmentStrategy_2','RecruitmentStrategy_3','Skill_Experience_Interaction'])
        preview_df_display['Status'] = preview_df_display['Status'].apply(
            lambda x: f"<span style='color:green;font-weight:bold;'>{x}</span>" if x=="Diterima"
                      else f"<span style='color:red;font-weight:bold;'>{x}</span>"
        )
        st.markdown(preview_df_display.to_html(escape=False, index=False), unsafe_allow_html=True)

        csv = df_pred.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download Hasil Prediksi", data=csv, file_name="hasil_prediksi.csv", mime="text/csv")

        top10 = df_pred.sort_values(by="Probability", ascending=False).head(10).copy()
        top10_display = top10[['Name','EducationLevel','ExperienceYears','RecruitmentStrategy','TotalScore','Probability','Status']]
        top10_display['Status'] = top10_display['Status'].apply(
            lambda x: f"<span style='color:green;font-weight:bold;'>{x}</span>" if x=="Diterima"
                      else f"<span style='color:red;font-weight:bold;'>{x}</span>"
        )
        st.markdown("<h3 style='margin-top:20px;'>Top 10 Kandidat Berdasarkan Probability</h3>", unsafe_allow_html=True)
        st.markdown(top10_display.to_html(escape=False, index=False), unsafe_allow_html=True)

    else:
        st.info("⚠ Belum ada data prediksi. Silakan lakukan upload dan prediksi di halaman Input Data.")

# ===================== Halaman EDA =====================
elif page == "EDA":
    st.markdown("<h2 style='text-align:center; background:linear-gradient(to right,#2b1e77,#6a0dad); color:white; padding:12px; border-radius:12px;'><i class='fas fa-chart-pie'></i> Exploratory Data Analysis</h2>", unsafe_allow_html=True)

    if "pred_df" in st.session_state and "df_input" in st.session_state:
        df_eda_pred = st.session_state.pred_df.copy()
        df_eda_orig = st.session_state.df_input.copy()
        df_eda_orig['Probability'] = df_eda_pred['Probability']
        df_eda_orig['Status'] = df_eda_pred['Status']

        # Pie Charts
        status_counts = df_eda_orig['Status'].value_counts().reset_index()
        status_counts.columns = ['Status','Count']
        fig_status = px.pie(status_counts, names='Status', values='Count',
                            color='Status',
                            color_discrete_map={'Diterima':'#4682B4','Tidak Diterima':'#89CFF0'},
                            title='Distribusi Status Kandidat')
        st.plotly_chart(fig_status)

        rec_counts = df_eda_orig['RecruitmentStrategy'].map({1:'Referral',2:'Job Fair',3:'Outsourcing'}).value_counts().reset_index()
        rec_counts.columns = ['RecruitmentStrategy','Count']
        fig_rec = px.pie(rec_counts, names='RecruitmentStrategy', values='Count',
                         color_discrete_sequence=["#336289",'#4682B4','#89CFF0'],
                         title='Distribusi Recruitment Strategy')
        st.plotly_chart(fig_rec)

        edu_counts = df_eda_orig['EducationLevel'].map({1:'High School',2:'Bachelor',3:'Master',4:'PhD'}).value_counts().reset_index()
        edu_counts.columns = ['EducationLevel','Count']
        fig_edu = px.pie(edu_counts, names='EducationLevel', values='Count',
                         color_discrete_sequence=["#284F6E", "#336289",'#4682B4','#89CFF0'],
                         title='Distribusi Education Level')
        st.plotly_chart(fig_edu)

        # Histogram numeric asli
        numeric_cols = ['ExperienceYears','InterviewScore','SkillScore','PersonalityScore']
        df_eda_orig['TotalScore'] = df_eda_orig['SkillScore'] + df_eda_orig['InterviewScore'] + df_eda_orig['PersonalityScore']
        numeric_cols.append('TotalScore')
        for col in numeric_cols:
            fig_bar = px.histogram(df_eda_orig, x=col, color_discrete_sequence=['#4682B4'], title=f'Distribusi {col}')
            st.plotly_chart(fig_bar)

    else:
        st.info("⚠ Belum ada data untuk EDA. Silakan lakukan prediksi terlebih dahulu di halaman Input Data.")
