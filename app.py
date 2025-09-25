import os
from joblib import load
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ===================== Page Config =====================
st.set_page_config(page_title="Recruitment Prediction App", layout="wide")
ACCENT_COLOR = "#1F3A93"
HISTORY_FILE = "history.csv"

# ===================== Load Font Awesome =====================
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
""", unsafe_allow_html=True)

# ===================== CSS Custom =====================
st.markdown("""
<style>
/* ===== Body ===== */
body {
    font-family: 'Arial', sans-serif;
}

/* ===== Topbar ===== */
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
.topbar .icon {
    font-size: 30px;
    color: white;
}

/* ===== Hero Title ===== */
.hero-title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: white;
    background: linear-gradient(to right, #2b1e77, #6a0dad);
    padding: 20px;
    border-radius: 18px;
    margin: 25px auto;
    width: 85%;
}
.hero-title i {
    margin-right: 10px;
}

/* ===== Instruction Box ===== */
.instruction-box {
    background: linear-gradient(to right, #2b1e77, #6a0dad);
    color: white;
    padding: 15px;
    border-radius: 12px;
    font-size: 20px;
    margin: 15px 0;
}

/* ===== Buttons ===== */
.stButton>button {
    background: linear-gradient(to right, #6a0dad, #2b1e77);
    color: white;
    font-weight: 900;
    border-radius: 25px;
    padding: 10px 24px;
    border: none;
    font-size: 28px;
    cursor: pointer;
    width: 100%;
}
.stButton>button:hover {
    opacity: 0.9;
}

/* ===== Result Box ===== */
.result-box {
    background: linear-gradient(to right, #2b1e77, #6a0dad);
    color: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    margin: 15px 0;
    font-weight: bold;
}

/* ===== Input Form ===== */
label, .stMarkdown p {
    font-size: 18px !important;
    font-weight: bold;
}
.stTextInput>div>div>input,
.stNumberInput>div>div>input,
.stSelectbox>div>div,
.stSlider>div>div {
    font-size: 18px !important;
    height: 45px !important;
}
.stNumberInput button {
    font-size: 18px !important;
    height: 45px !important;
    width: 45px !important;
}

/* ===== Table Riwayat ===== */
.tbl-header {
    background-color: #E6E6FA;
    font-weight: bold;
    text-align: center;
    border: 1px solid #ccc;
    padding: 10px;
}
.tbl-cell {
    border: 1px solid #ccc;
    padding: 6px;
    text-align: center;
    font-size: 16px;
}

/* ===== Delete Button Icon ===== */
.delete-btn {
    background: transparent;
    color: red;
    border: none;
    cursor: pointer;
    font-size: 18px;
    padding: 0;
}
.delete-btn:hover {
    color: darkred;
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

st.markdown("""
<p style="text-align:center; font-size:18px; font-weight:bold;">
Aplikasi untuk membantu perusahaan menyeleksi kandidat lebih cepat, objektif,
dan efisien, sehingga menghemat waktu serta biaya dalam pengambilan keputusan.
</p>
""", unsafe_allow_html=True)

# ===================== Load Model & Scaler =====================
model, scaler = None, None
if os.path.exists("best_catboost_optuna.joblib") and os.path.exists("scaler.joblib"):
    model = joblib.load("best_catboost_optuna.joblib")
    scaler = joblib.load("scaler.joblib")
else:
    st.warning("⚠ File model atau scaler tidak ditemukan. Prediksi tidak bisa dijalankan.")

# ===================== Preprocessing Function =====================
def preprocess_input(data, scaler):
    numeric_columns = ['ExperienceYears', 'InterviewScore', 'SkillScore', 'PersonalityScore']
    df = pd.DataFrame([data])
    recruitment_strategy_map = {'Referral':[1,0,0],'Job Fair':[0,1,0],'Outsourcing':[0,0,1]}
    recruitment_cols = ['RecruitmentStrategy_1','RecruitmentStrategy_2','RecruitmentStrategy_3']
    df[recruitment_cols] = recruitment_strategy_map[data['RecruitmentStrategy']]
    df['TotalScore'] = df['SkillScore'] + df['InterviewScore'] + df['PersonalityScore']
    df['Skill_Experience_Interaction'] = df['SkillScore'] * df['ExperienceYears']
    df[numeric_columns] = scaler.transform(df[numeric_columns])
    selected_features = [
        'EducationLevel','ExperienceYears','InterviewScore','SkillScore',
        'PersonalityScore','RecruitmentStrategy_1','RecruitmentStrategy_2',
        'TotalScore','Skill_Experience_Interaction','RecruitmentStrategy_3'
    ]
    return df[selected_features]

# ===================== Load & Save History =====================
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            if df.empty or df.columns.size == 0: return []
            return df.to_dict("records")
        except pd.errors.EmptyDataError:
            return []
    else:
        return []

def save_history():
    pd.DataFrame(st.session_state.history).to_csv(HISTORY_FILE, index=False)

# ===================== Session State =====================
if "page" not in st.session_state: st.session_state.page = "main"
if "history" not in st.session_state: st.session_state.history = load_history()

# ===================== Halaman Utama =====================
if st.session_state.page == "main":
    st.markdown("<h2 style='text-align:center;'>📝 Input Data Kandidat</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        nama = st.text_input("Nama Kandidat")
        education_level = st.selectbox("Education Level", [1,2,3,4],
            format_func=lambda x: {1:"High School",2:"Bachelor",3:"Master",4:"PhD"}[x])
        experience_years = st.number_input("Experience Years", 0, 40)
        recruitment_strategy = st.selectbox("Recruitment Strategy", ['Referral','Job Fair','Outsourcing'])
        interview_score = st.slider("Interview Score", 0, 100, 50)
        skill_score = st.slider("Skill Score", 0, 100, 50)
        personality_score = st.slider("Personality Score", 0, 100, 50)
    with col2:
        st.markdown("""
        <div class="instruction-box">
            <b>Petunjuk:</b><br>
            - Isi semua data kandidat.<br>
            - Skor: 0 (rendah) — 100 (tinggi).<br>
            - Pilih strategi rekrutmen.<br>
            - Klik <b>Prediksi</b> untuk melihat hasil.<br>
        </div>
        """, unsafe_allow_html=True)

    col_pred, col_hist = st.columns([1,1])
    with col_pred:
        prediksi_btn = st.button("Prediksi")
    with col_hist:
        if st.button("Riwayat"):
            st.session_state.page = "riwayat"

    if prediksi_btn and model and scaler:
        input_data = {
            'EducationLevel': education_level,
            'ExperienceYears': experience_years,
            'InterviewScore': interview_score,
            'SkillScore': skill_score,
            'PersonalityScore': personality_score,
            'RecruitmentStrategy': recruitment_strategy
        }
        processed_input = preprocess_input(input_data, scaler)
        prediction_proba = model.predict_proba(processed_input)[0]
        prob_accept = prediction_proba[1] * 100
        threshold = 61.0
        prediction = 1 if prob_accept >= threshold else 0
        st.session_state.history.append({
            "Nama": nama,
            "Probabilitas": prob_accept,
            "Status": "Diterima" if prediction == 1 else "Tidak Diterima"
        })
        save_history()

        st.markdown(f"""
        <div style="text-align:center; background:linear-gradient(to right,#2b1e77,#6a0dad);
                    color:white; padding:14px; border-radius:12px; font-size:28px; font-weight:bold;">
            <i class="fas fa-chart-line"></i> Hasil Prediksi
        </div>
        """, unsafe_allow_html=True)

        col_left, col_right = st.columns([1,2])
        with col_left:
            st.markdown(f"""
            <div class="result-box" style="font-size:22px;">
                <b>Probabilitas Diterima</b><br>{prob_accept:.2f}%
            </div>
            """, unsafe_allow_html=True)
            if prediction == 1:
                st.markdown("""<div class="result-box" style="font-size:22px; color:lightgreen;">✅ Diterima</div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="result-box" style="font-size:22px; color:#ff6b6b;">❌ Tidak Diterima</div>""", unsafe_allow_html=True)
        with col_right:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_accept,
                title={'text':"Probabilitas Diterima",'font':{'size':22}},
                number={'valueformat':'.2f','suffix':"%",'font':{'size':42}},
                gauge={'axis':{'range':[0,100]},
                       'bar':{'color':ACCENT_COLOR},
                       'steps':[{'range':[0,61],'color':"#FADBD8"},{'range':[61,100],'color':"#D8F4D7"}]}
            ))
            fig.update_layout(height=550, width=850)
            st.plotly_chart(fig, use_container_width=True)

# ===================== Halaman Riwayat =====================
elif st.session_state.page == "riwayat":
    st.markdown("""
    <div style="text-align:center; background:linear-gradient(to right,#2b1e77,#6a0dad);
                color:white; padding:14px; border-radius:12px; font-size:28px; font-weight:bold; margin-bottom:10px;">
        <i class="fas fa-history"></i> Riwayat
    </div>
    """, unsafe_allow_html=True)

    # Tabel Riwayat
    if len(st.session_state.history) == 0:
        st.info("📭 Belum ada riwayat prediksi.")
    else:
        df_history = pd.DataFrame(st.session_state.history).sort_values(by="Probabilitas", ascending=False).reset_index(drop=True)
        header_cols = st.columns([3,2,2,0.5])
        header_cols[0].markdown("<div class='tbl-header'>Nama</div>", unsafe_allow_html=True)
        header_cols[1].markdown("<div class='tbl-header'>Probabilitas</div>", unsafe_allow_html=True)
        header_cols[2].markdown("<div class='tbl-header'>Status</div>", unsafe_allow_html=True)
        header_cols[3].markdown("", unsafe_allow_html=True)

        for i, row in df_history.iterrows():
            col1, col2, col3, col4 = st.columns([3,2,2,0.5])
            col1.markdown(f"<div class='tbl-cell'>{row['Nama']}</div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='tbl-cell'>{row['Probabilitas']:.2f}%</div>", unsafe_allow_html=True)
            col3.markdown(f"<div class='tbl-cell'>{row['Status']}</div>", unsafe_allow_html=True)
            with col4:
                if st.button("🗑", key=f"hapus_{i}"):
                    st.session_state.history.pop(i)
                    save_history()

        # Tombol Download
        df_download = pd.DataFrame(st.session_state.history)
        csv = df_download.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download History", data=csv, file_name="history.csv", mime="text/csv")

    # Tombol Back & Clear
    col_back, col_mid, col_clear = st.columns([1,13,1])
    with col_back:
        if st.button("Back"):
            st.session_state.page = "main"
    with col_clear:
        if st.button("Clear"):
            st.session_state.history = []
            save_history()

