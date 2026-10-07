import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import time

# Set page config
st.set_page_config(
    page_title="Sistem Klasifikasi Sentimen",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load model dan tokenizer
@st.cache_resource
def load_artifacts():
    try:
        model = load_model("final_sentiment_model.h5")
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Fungsi cleaning text
def clean_indonesian_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Fungsi prediksi
def predict_sentiment(text):
    text_clean = clean_indonesian_text(text)
    seq = tokenizer.texts_to_sequences([text_clean])
    pad = pad_sequences(seq, maxlen=100, padding="post", truncating="post")
    prob = model.predict(pad, verbose=0)[0][0]
    
    if prob > 0.5:
        label = "Positif"
        confidence = float(prob)
    else:
        label = "Negatif" 
        confidence = float(1 - prob)
    
    return label, confidence

# Custom CSS untuk dark theme
st.markdown("""
<style>
    /* Background gradient dark */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%) !important;
        min-height: 100vh;
        color: white;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .welcome-header {
        text-align: center;
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .main-title {
        text-align: center;
        color: white;
        font-size: 2.2rem;
        font-weight: bold;
        margin: 2rem 0 1rem 0;
    }
    
    .subtitle {
        text-align: center;
        color: #d1d5db;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    /* Glass card effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 1rem;
    }
    
    /* Feature cards styling */
    .feature-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 0.75rem;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        border-color: rgba(99, 102, 241, 0.5);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-title {
        color: white;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        color: #9ca3af;
        font-size: 0.95rem;
        line-height: 1.4;
    }
    
    /* Input section styling */
    .input-section {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 1rem;
        padding: 2rem;
        margin: 2rem 0;
    }
    
    .input-label {
        color: white;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    
    /* Textarea styling */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 2px solid #4b5563 !important;
        border-radius: 0.75rem !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        color: white !important;
        resize: vertical;
    }
    
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.3) !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #9ca3af !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.75rem !important;
        padding: 1rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.4) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.5) !important;
    }
    
    /* Result card styling */
    .result-card {
        border-radius: 1rem;
        padding: 2rem;
        color: white;
        margin-top: 2rem;
        animation: slideUp 0.4s ease-out forwards;
        opacity: 0;
        transform: translateY(20px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    @keyframes slideUp {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .positive-gradient {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    
    .negative-gradient {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }
    
    .confidence-bar {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 9999px;
        height: 12px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .confidence-fill {
        background: white;
        height: 100%;
        border-radius: 9999px;
        transition: width 1s ease-out;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    
    /* Status indicator */
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        justify-content: center;
        margin-bottom: 2rem;
    }
    
    .pulse-dot {
        width: 12px;
        height: 12px;
        background-color: #10b981;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Hero icon */
    .hero-icon {
        animation: float 3s ease-in-out infinite;
        margin: 2rem auto;
        text-align: center;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .hero-emoji {
        width: 80px;
        height: 80px;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        border-radius: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.5rem;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);
        margin: 0 auto;
    }
    
    /* Expandable section styling */
    .expandable-section {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 0.75rem;
        margin-top: 1.5rem;
        animation: fadeIn 0.5s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .detail-item {
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .detail-item:last-child {
        border-bottom: none;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Divider styling */
    .divider {
        height: 1px;
        background: rgba(255, 255, 255, 0.1);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main App
def main():
    # Load model
    global model, tokenizer
    model, tokenizer = load_artifacts()
    
    if model is None:
        st.error("Gagal memuat model. Pastikan file 'final_sentiment_model.h5' dan 'tokenizer.pkl' ada di folder yang sama.")
        return

    # Navigation/Status Bar
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="status-indicator">
            <div class="pulse-dot"></div>
            <span style="color: #d1d5db;">Sistem Aktif</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Header Section
    st.markdown('<div class="welcome-header">Sistem Klasifikasi Sentimen</div>', unsafe_allow_html=True)
    
    # Divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Hero Section dengan icon
    st.markdown("""
    <div class="hero-icon">
        <div class="hero-emoji">🔍</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Title
    st.markdown('<div class="main-title">Analisis Sentimen Cerdas</div>', unsafe_allow_html=True)
    
    # Subtitle
    st.markdown('<div class="subtitle">Teknologi Deep Learning BiLSTM untuk mengklasifikasikan sentimen teks Bahasa Indonesia dengan akurasi tinggi</div>', unsafe_allow_html=True)
    
    # Divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Features Section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Cepat & Akurat</div>
            <div class="feature-desc">Hasil analisis instan dengan tingkat akurasi tinggi</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">BiLSTM Model</div>
            <div class="feature-desc">Menggunakan teknologi deep learning terkini</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🇮🇩</div>
            <div class="feature-title">Bahasa Indonesia</div>
            <div class="feature-desc">Dioptimalkan khusus untuk teks berbahasa Indonesia</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Input Section dengan glass card
    st.markdown("""
    <div class="glass-card" style="padding: 2rem;">
        <div class="input-label">✍️ Masukkan Teks untuk Dianalisis</div>
    """, unsafe_allow_html=True)
    
    text_input = st.text_area(
        "",
        placeholder="Contoh: Saya sangat senang dengan layanan ini! Pelayanannya luar biasa.",
        height=150,
        label_visibility="collapsed"
    )
    
    # Predict Button
    predict_btn = st.button("**🚀 Analisis Sentimen Sekarang**")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Results Section
    if predict_btn:
        if text_input.strip():
            # Show loading state
            with st.spinner(""):
                loading_placeholder = st.empty()
                loading_placeholder.markdown(
                    '<div style="text-align: center; padding: 2rem;"><div style="font-size: 1.25rem; color: #9ca3af;">⏳ Sedang Menganalisis...</div></div>', 
                    unsafe_allow_html=True
                )
                
                # Simulate loading delay
                time.sleep(1.5)
                
                # Get prediction
                label, confidence = predict_sentiment(text_input)
                confidence_percent = confidence * 100
                cleaned_text = clean_indonesian_text(text_input)  # Simpan hasil cleaning
                
                # Determine styling based on sentiment
                if label == "Positif":
                    gradient_class = "result-card positive-gradient"
                    emoji = "😊"
                else:
                    gradient_class = "result-card negative-gradient"
                    emoji = "😞"
                
                # Clear loading
                loading_placeholder.empty()
                
                # Display result
                st.markdown(f"""
                <div class="{gradient_class}">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.5rem;">
                        <div>
                            <p style="margin: 0; font-size: 0.875rem; opacity: 0.8; margin-bottom: 0.5rem;">Hasil Klasifikasi</p>
                            <h3 style="margin: 0; font-size: 2rem; font-weight: bold;">Sentimen {label}</h3>
                        </div>
                        <span style="font-size: 4rem;">{emoji}</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence_percent}%;"></div>
                    </div>
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <p style="margin: 0; font-size: 0.875rem; opacity: 0.9;">Tingkat Keyakinan</p>
                        <p style="margin: 0; font-size: 1.5rem; font-weight: bold;">{confidence_percent:.1f}%</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Expandable Detail Section - FIXED VERSION
                with st.expander("ℹ️ Informasi Detail", expanded=False):
                    st.markdown(f"""
                    <div class="expandable-section" style="padding: 1.5rem;">
                        <div class="detail-item">
                            <strong>📝 Teks Input:</strong><br>
                            <span style="color: #d1d5db;">'{text_input}'</span>
                        </div>
                        <div class="detail-item">
                            <strong>🧹 Teks setelah cleaning:</strong><br>
                            <span style="color: #d1d5db;">'{cleaned_text}'</span>
                        </div>
                        <div class="detail-item">
                            <strong>🎯 Confidence score:</strong><br>
                            <span style="color: #d1d5db;">{confidence:.4f}</span>
                        </div>
                        <div class="detail-item">
                            <strong>🏷️ Kategori:</strong><br>
                            <span style="color: #d1d5db;">{label}</span>
                        </div>
                        <div class="detail-item">
                            <strong>🤖 Model:</strong><br>
                            <span style="color: #d1d5db;">BiLSTM Neural Network</span>
                        </div>
                        <div class="detail-item">
                            <strong>📊 Akurasi Model:</strong><br>
                            <span style="color: #d1d5db;">~86%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                        
        else:
            st.warning("⚠️ Silakan masukkan teks terlebih dahulu!")

if __name__ == "__main__":
    main()