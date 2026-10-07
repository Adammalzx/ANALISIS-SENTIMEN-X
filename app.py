from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model
from keras.utils import pad_sequences
import pickle  # <--- Cukup begini saja
import re

# Inisialisasi Aplikasi Flask
app = Flask(__name__)

# --- 1. LOAD MODEL & TOKENIZER (Hanya sekali saat server nyala) ---
print("⏳ Sedang memuat model BiLSTM & Tokenizer...")
try:
    # Load Model
    model = load_model("final_sentiment_model.h5")
    
    # Load Tokenizer
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
        
    print("✅ Model & Tokenizer berhasil dimuat! Siap prediksi.")
except Exception as e:
    print(f"❌ Error Critical: Gagal memuat model/tokenizer. {e}")
    model = None
    tokenizer = None

# --- 2. FUNGSI CLEANING (Sama persis dengan Streamlit) ---
def clean_indonesian_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 3. API ENDPOINT (Pintu Masuk dari Android) ---
@app.route('/api/prediksi', methods=['POST'])
def prediksi_sentimen():
    print("📦 RAW BODY:", request.data)
    data_masuk = request.get_json(force=True, silent=True)
    print("📦 PARSED JSON:", data_masuk)
    # Cek apakah model sudah siap
    if model is None or tokenizer is None:
        return jsonify({
            'status': 'error',
            'pesan': 'Model belum siap di server.'
        }), 500

    try:
        # Ambil data JSON yang dikirim Android
        data_masuk = request.get_json(force=True, silent=True)
        if not data_masuk or 'teks' not in data_masuk:
            return jsonify({'status': 'error', 'pesan': 'Data tidak valid/kosong'}), 400
        
        raw_text = data_masuk['teks']
        
        # --- PROSES (Logic dari Streamlit dipindah ke sini) ---
        
        # 1. Cleaning
        text_clean = clean_indonesian_text(raw_text)
        
        # 2. Tokenizing & Padding
        seq = tokenizer.texts_to_sequences([text_clean])
        pad = pad_sequences(seq, maxlen=100, padding="post", truncating="post")
        
        # 3. Prediksi
        prob = model.predict(pad, verbose=0)[0][0]
        
        # 4. Tentukan Label
        if prob > 0.5:
            label = "Negatif"
            confidence = float(prob)
        else:
            label = "Positif"
            confidence = float(1 - prob)
            
        # Format persentase keyakinan
        confidence_percent = f"{confidence * 100:.1f}%"

        print(f"📩 Input: {raw_text} | 🧼 Clean: {text_clean} | 🤖 Hasil: {label} ({confidence_percent})")

        # --- KIRIM BALIK KE ANDROID ---
        return jsonify({
            'status': 'sukses',
            'sentimen': label,
            'confidence': confidence_percent,
            'confidence_score': confidence,
            'teks_clean': text_clean
        })

    except Exception as e:
        print(f"❌ Error saat prediksi: {e}")
        return jsonify({'status': 'error', 'pesan': str(e)}), 500

if __name__ == '__main__':
    # Jalankan server di IP 0.0.0.0 supaya bisa diakses HP
    # Port 5000 adalah standar Flask
    app.run(host='0.0.0.0', port=5000, debug=True)