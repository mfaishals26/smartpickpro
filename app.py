import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time
import math

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Smartpick Pro",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. SESSION STATE (INIT) ---
if 'search_results' not in st.session_state: st.session_state.search_results = None
if 'page_number' not in st.session_state: st.session_state.page_number = 1
if 'prediksi_kat' not in st.session_state: st.session_state.prediksi_kat = ""
# Flag Balon
if 'trigger_balloons' not in st.session_state: st.session_state.trigger_balloons = False

# --- 3. LOAD DATA & MODEL ---
KURS_EUR_IDR = 17000 

@st.cache_data
def load_data():
    try:
        # Pastikan file ini ada di GitHub/Folder yang sama
        df = pd.read_csv('gsm_cleaned_final.csv')
        
        # Buat kategori jika belum ada
        def categorize(price):
            if price < 200: return 'Budget'
            elif price < 400: return 'Mid-Range'
            elif price < 700: return 'High-End'
            else: return 'Flagship'
            
        if 'price_category' not in df.columns:
            df['price_category'] = df['price_eur'].apply(categorize)

        # Filter Data
        df = df[df['ram_gb'] >= 3]
        df = df[df['storage_gb'] >= 32]
        df = df[~( (df['price_eur'] > 300) & (df['ram_gb'] < 4) )]
        
        return df
    except: return None

@st.cache_resource
def train_model(df):
    X = df[['ram_gb', 'storage_gb', 'battery_mah']]
    y = df['price_category']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

df = load_data()
if df is None: st.stop()
rf_model = train_model(df)

# --- 4. FUNGSI CALLBACK (LOGIKA PENCARIAN) ---
def proses_pencarian():
    budget_val = st.session_state.inp_budget
    ram_val = st.session_state.inp_ram
    storage_val = st.session_state.inp_storage
    bat_val = st.session_state.inp_bat
    
    # 1. Prediksi AI
    user_input = pd.DataFrame([[ram_val, storage_val, bat_val]], columns=['ram_gb', 'storage_gb', 'battery_mah'])
    prediksi = rf_model.predict(user_input)[0]
    st.session_state.prediksi_kat = prediksi
    
    # 2. Filter
    budget_eur = budget_val / KURS_EUR_IDR
    candidates = df[df['price_eur'] <= budget_eur].copy()
    
    if candidates.empty:
        st.session_state.search_results = pd.DataFrame()
        st.session_state.trigger_balloons = False
    else:
        # 3. Hitung Score
        candidates['score'] = np.sqrt(
            (candidates['ram_gb'] - ram_val)**2 * 1.5 + 
            (candidates['storage_gb'] - storage_val)**2 * 1.0 + 
            ((candidates['battery_mah'] - bat_val)/100)**2 * 0.5
        )
        
        # 4. Sorting Prioritas
        candidates['is_priority'] = candidates['price_category'] == prediksi
        
        st.session_state.search_results = candidates.sort_values(
            by=['is_priority', 'score'], 
            ascending=[False, True]
        )
        st.session_state.trigger_balloons = True
        
    st.session_state.page_number = 1

# --- 5. HELPER FUNCTIONS ---
def get_icon_url(): return "https://cdn-icons-png.flaticon.com/512/644/644458.png"
def make_link(brand, model): return f"https://www.google.com/search?q=gsmarena+{brand}+{model}".replace(' ', '+')
def render_bar(value, max_val, color_class):
    pct = min((value / max_val) * 100, 100)
    return f"""<div class="bar-container"><div class="{color_class}" style="width: {pct}%;"></div></div>"""

# --- 6. CSS STYLING ---
st.markdown("""
<style>
    .stApp { background: radial-gradient(circle at 10% 20%, rgb(25, 30, 40) 0%, rgb(0, 0, 0) 90%); font-family: 'Inter', sans-serif; color: white; }
    
    .gradient-text { background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3.5rem; font-weight: 900; text-align: center; letter-spacing: -1px; margin-bottom: 5px; }
    .subtitle-text { text-align: center; color: #aaa; margin-bottom: 40px; font-weight: 300; font-size: 1.1rem; }
    
    .stSlider label, .stSelectbox label, .stNumberInput label, .stMarkdown p { color: #e0e0e0 !important; font-weight: 600; font-size: 1rem; }
    div[data-testid="stThumbValue"] { color: #ffffff !important; background-color: transparent !important; }
    div[data-testid="stTickBarMin"], div[data-testid="stTickBarMax"] { color: #aaaaaa !important; }

    .best-card { background: white; border-radius: 25px; padding: 30px; box-shadow: 0 0 50px rgba(0, 198, 255, 0.3); border: 2px solid #00c6ff; position: relative; overflow: hidden; color: #333; }
    .best-ribbon { position: absolute; top: 0; right: 0; background: linear-gradient(90deg, #00c6ff, #0072ff); color: white; padding: 5px 20px; border-bottom-left-radius: 20px; font-weight: bold; }
    
    .grid-container { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-top: 20px; }
    .phone-card { background: white; border-radius: 15px; padding: 15px; text-align: center; transition: all 0.2s; border: 1px solid rgba(0,0,0,0.1); height: 100%; color: #333; display: flex; flex-direction: column; justify-content: space-between; }
    .phone-card:hover { transform: translateY(-5px); box-shadow: 0 5px 15px rgba(255,255,255,0.2); }
    
    .card-brand { font-size: 0.75rem; color: #888; font-weight: 700; text-transform: uppercase; margin-bottom: 5px; }
    .card-model { font-weight: 700; color: #333; font-size: 1rem; margin-bottom: 5px; line-height: 1.2; }
    .card-price { color: #0072ff; font-weight: 800; font-size: 1rem; margin-bottom: 10px; }
    .card-spec { text-align: left; font-size: 0.7rem; color: #555; }

    @media only screen and (max-width: 600px) {
        .grid-container { grid-template-columns: repeat(2, 1fr) !important; gap: 10px !important; }
        .phone-card { padding: 10px !important; border-radius: 12px !important; }
        .card-model { font-size: 0.85rem !important; height: 40px; overflow: hidden; text-overflow: ellipsis; }
        .card-price { font-size: 0.9rem !important; }
        .card-spec { font-size: 0.6rem !important; }
        .gradient-text { font-size: 2.5rem !important; }
        .subtitle-text { font-size: 0.9rem !important; }
        .hero-flex { flex-direction: column !important; text-align: center !important; }
        .hero-grid-spec { grid-template-columns: 1fr !important; }
    }

    .bar-container { background-color: #e0e0e0; border-radius: 10px; height: 5px; width: 100%; margin: 3px 0; overflow: hidden; }
    .bar-fill-ram { height: 100%; background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%); }
    .bar-fill-bat { height: 100%; background: linear-gradient(90deg, #fce38a 0%, #f38181 100%); }
    a.card-link { text-decoration: none; color: inherit; display: block; height: 100%; }
    .stat-box { text-align: center; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 15px; border: 1px solid rgba(255,255,255,0.2); }
    .stat-num { font-size: 1.5rem; font-weight: bold; color: white; }
    .stat-label { font-size: 0.8rem; color: #ccc; text-transform: uppercase; }
    
    div.stButton > button { width: 100%; border-radius: 12px; height: 50px; font-weight: 700; transition: 0.3s; background: linear-gradient(90deg, #00c6ff, #0072ff); border: none; color: white; }
    .footer { text-align: center; padding: 30px; color: #888; font-size: 0.9rem; margin-top: 50px; border-top: 1px solid rgba(255,255,255,0.1); }
</style>
""", unsafe_allow_html=True)

# --- 7. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.title("Panduan")
    st.info("**Cara Pakai:**\n1. Atur Budget.\n2. Pilih RAM & Storage.\n3. Tentukan Baterai.\n4. Klik Cari.")
    st.divider()
    st.markdown("**Created with ❤️ by FAISHAL**")

# --- 8. UI INPUT ---
st.markdown("<h1 class='gradient-text'>Smartpick Pro</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Temukan Smartphone Impian dengan Analisis Cerdas</p>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1: st.slider("Geser untuk budget maksimal", 1_000_000, 30_000_000, 5_000_000, step=500_000, format="Rp %d", key='inp_budget')
with c2: 
    cc1, cc2 = st.columns(2)
    with cc1: st.selectbox("Kapasitas RAM", [3,4,6,8,12,16,24], index=3, format_func=lambda x: f"{x} GB", key='inp_ram')
    with cc2: st.selectbox("Memori Internal", [32,64,128,256,512,1024], index=3, format_func=lambda x: f"{x} GB", key='inp_storage')
with c3: st.slider("Kapasitas (mAh)", 3000, 7000, 5000, step=100, key='inp_bat')

st.write("")
col_btn1, col_btn2, col_btn3 = st.columns([2,1,2])
with col_btn2:
    st.button("✨ CARI SEKARANG ✨", type="primary", on_click=proses_pencarian)

st.markdown("---") 

# --- 9. DISPLAY HASIL ---
if st.session_state.search_results is not None:
    results = st.session_state.search_results
    
    if results.empty:
        st.error("❌ Tidak ditemukan HP yang sesuai. Coba naikkan Budget.")
    else:
        # Balon
        if st.session_state.trigger_balloons:
            st.balloons()
            st.session_state.trigger_balloons = False

        # Dashboard
        st.markdown('<div style="margin-bottom: 30px;">', unsafe_allow_html=True)
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1: st.markdown(f"<div class='stat-box'><div class='stat-num'>{len(results)}</div><div class='stat-label'>HP Ditemukan</div></div>", unsafe_allow_html=True)
        with sc2: st.markdown(f"<div class='stat-box'><div class='stat-num'>{st.session_state.prediksi_kat}</div><div class='stat-label'>Kelas Prediksi</div></div>", unsafe_allow_html=True)
        with sc3: st.markdown(f"<div class='stat-box'><div class='stat-num'>Rp {(results['price_eur'].min() * KURS_EUR_IDR)/1000000:.1f}jt</div><div class='stat-label'>Termurah</div></div>", unsafe_allow_html=True)
        with sc4: st.markdown(f"<div class='stat-box'><div class='stat-num'>Rp {(results['price_eur'].max() * KURS_EUR_IDR)/1000000:.1f}jt</div><div class='stat-label'>Termahal</div></div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        items_per_page = 12
        best_phone = results.iloc[0]
        others = results.iloc[1:]
        total_pages = math.ceil(len(others) / items_per_page)

        # HERO SECTION
        if st.session_state.page_number == 1:
            best_price = best_phone['price_eur'] * KURS_EUR_IDR
            link_hero = make_link(best_phone['brand'], best_phone['model'])
            ram_bar = render_bar(best_phone['ram_gb'], 24, "bar-fill-ram")
            bat_bar = render_bar(best_phone['battery_mah'], 7000, "bar-fill-bat")
            
            badge_text = f"🏆 TOP PICK {st.session_state.prediksi_kat.upper()}"
            if best_phone['price_category'] != st.session_state.prediksi_kat:
                badge_text = "⭐ ALTERNATIF TERBAIK"

            st.markdown(f"""
            <a href="{link_hero}" target="_blank" class="card-link">
                <div class="best-card">
                    <div class="best-ribbon">{badge_text}</div>
                    <div class="hero-flex" style="display:flex; flex-wrap:wrap; align-items:center; gap:30px;">
                        <div class="hero-img-container" style="flex:1; text-align:center;">
                            <img src="{get_icon_url()}" style="width:160px; opacity:0.9;">
                        </div>
                        <div class="hero-text-container" style="flex:2;">
                            <span style="background:#000; color:#fff; padding:3px 8px; border-radius:5px; font-size:0.8rem; letter-spacing:1px;">{best_phone['brand'].upper()}</span>
                            <h1 style="margin:5px 0; color:#333;">{best_phone['model']}</h1>
                            <h2 style="color:#0072ff; margin-bottom:15px;">Rp {best_price:,.0f}</h2>
                            <div class="hero-grid-spec" style="display:grid; grid-template-columns: 1fr 1fr; gap:20px;">
                                <div><b>RAM {best_phone['ram_gb']}GB</b> {ram_bar}</div>
                                <div><b>BAT {best_phone['battery_mah']}mAh</b> {bat_bar}</div>
                            </div>
                        </div>
                    </div>
                </div>
            </a>
            """, unsafe_allow_html=True)

        # GRID SECTION
        start_idx = (st.session_state.page_number - 1) * items_per_page
        end_idx = start_idx + items_per_page
        current_page = others.iloc[start_idx:end_idx]

        if not current_page.empty:
            st.subheader(f"📱 Halaman {st.session_state.page_number}")
            
            html_content = '<div class="grid-container">'
            for idx, row in current_page.reset_index().iterrows():
                p_idr = row['price_eur'] * KURS_EUR_IDR
                lnk = make_link(row['brand'], row['model'])
                r_bar = render_bar(row['ram_gb'], 24, "bar-fill-ram")
                b_bar = render_bar(row['battery_mah'], 7000, "bar-fill-bat")
                
                # HTML Rata Kiri
                card_html = f"""
<a href="{lnk}" target="_blank" class="card-link">
<div class="phone-card">
<img src="{get_icon_url()}" style="width:40px; margin-bottom:5px; display:block; margin-left:auto; margin-right:auto;">
<div class="card-brand">{row['brand'].upper()}</div>
<div class="card-model">{row['model']}</div>
<div class="card-price">Rp {p_idr:,.0f}</div>
<div class="card-spec">
<div>💾 {row['ram_gb']}GB {r_bar}</div>
<div style="margin-top:2px;">🔋 {row['battery_mah']}mAh {b_bar}</div>
</div>
</div>
</a>
"""
                html_content += card_html
            html_content += '</div>'
            st.markdown(html_content, unsafe_allow_html=True)

        # PAGINATION
        st.markdown("<br>", unsafe_allow_html=True)
        col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
        with col_p1:
            if st.session_state.page_number > 1:
                if st.button("⬅️ Back"):
                    st.session_state.page_number -= 1
                    st.rerun()
        with col_p2:
            st.markdown(f"<div style='text-align:center; color:white; font-weight:bold; padding-top:10px;'>Page {st.session_state.page_number} / {total_pages}</div>", unsafe_allow_html=True)
        with col_p3:
            if st.session_state.page_number < total_pages:
                if st.button("Next ➡️"):
                    st.session_state.page_number += 1
                    st.rerun()

# --- 10. FOOTER ---
st.markdown("""
<div class="footer">
    Smartpick Pro © 2025 | Dibuat oleh FAISHAL<br>
    Powered by Streamlit & Random Forest Algorithm
</div>
""", unsafe_allow_html=True)

