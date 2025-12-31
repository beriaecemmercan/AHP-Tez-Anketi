# ahp.py
import os
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st

# ---- Yollar ----
# Masaüstündeki klasör yapına göre BASE yolunu ayarlar
BASE = os.path.join(os.path.expanduser("~"), "Desktop", "TEZ_KODLAR_SON")
RESP_DIR = os.path.join(BASE, "survey_responses")
os.makedirs(RESP_DIR, exist_ok=True)
RAW_CSV = os.path.join(RESP_DIR, "ahp_raw.csv")

# ---------------------------------------------------------------------
# ---- KRİTERLER VE PERFORMANS SKORLARI (Görseldeki Güncel Veriler) ----
# ---------------------------------------------------------------------
aspects_tr = {
    "Accuracy/Consistency": "Doğruluk/Bilgi Tutarlılığı",
    "Code & Development": "Kod & Geliştiricilik",
    "Overall Satisfaction": "Genel Memnuniyet",
    "Interface/Usability": "Arayüz/Kolaylık",
    "Creativity/Visual": "Yaratıcılık/Görsel",
    "System Performance/Uptime": "Sistem Performansı/Kesintisizlik",
    "Education/Learning": "Eğitim/Öğrenme",
    "Price/Cost": "Fiyat/Maliyet"
}

aspect_keys = list(aspects_tr.keys())
n = len(aspect_keys)

# Karar Matrisi: Görseldeki 5 Model x 8 Kriter (Satır: Model, Sütun: Kriter)
models = ["CHATGPT", "CLAUDE", "COPILOT", "GEMINI", "GROK"]
raw_matrix = np.array([
    [0.534583, 0.430275, 0.847357, 0.665037, 0.488741, 0.344586, 0.835551, 0.361533], # CHATGPT
    [0.606707, 0.643581, 0.829183, 0.659910, 0.600000, 0.472727, 0.722222, 0.361533], # CLAUDE
    [0.697545, 0.629730, 0.910492, 0.832819, 0.659259, 0.590580, 0.815493, 0.805134], # COPILOT
    [0.424970, 0.376968, 0.798503, 0.642006, 0.579021, 0.347259, 0.691142, 1.000000], # GEMINI
    [0.700422, 0.548165, 0.914431, 0.852204, 0.663430, 0.437500, 0.760081, 0.000000], # GROK
])

# ---- TOPSIS ALGORİTMASI ----
def run_topsis(matrix, weights):
    # 1. Adım: Normalizasyon (Vektör Normalizasyonu)
    norm_matrix = matrix / np.sqrt((matrix**2).sum(axis=0))
    # 2. Adım: Ağırlıklı Normalize Matris oluşturma
    weighted_matrix = norm_matrix * weights
    # 3. Adım: Pozitif ve Negatif İdeal Çözümlerin belirlenmesi
    ideal_best = np.max(weighted_matrix, axis=0)
    ideal_worst = np.min(weighted_matrix, axis=0)
    # 4. Adım: İdeal çözümlere olan Öklid mesafeleri
    dist_best = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))
    # 5. Adım: Göreceli yakınlık katsayısının hesaplanması
    scores = dist_worst / (dist_best + dist_worst)
    return scores

# ---- AHP MATEMATİKSEL FONKSİYONLAR ----
scale_labels = ["9", "8", "7", "6", "5", "4", "3", "2", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
scale_magnitudes = [9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def idx_to_saaty(idx: int) -> float:
    mag = scale_magnitudes[idx]
    return float(mag) if idx < 8 else (1.0 if idx == 8 else 1.0 / float(mag))

def ahp_weights(A):
    eigvals, eigvecs = np.linalg.eig(A)
    idx = np.argmax(eigvals.real)
    w = np.abs(eigvecs[:, idx].real)
    w = w / w.sum()
    lambda_max = eigvals[idx].real
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}.get(n, 1.45)
    CR = CI / RI if RI > 0 else 0.0
    return w, CR, lambda_max, CI

# ---- STREAMLIT ARAYÜZÜ ----
st.set_page_config(page_title="AI Seçim Aracı", page_icon="📈", layout="centered")
st.title("AHP & TOPSIS Tabanlı Model Önerisi")
st.markdown("Kriterleri önem sırasına göre kıyaslayın, size en uygun modeli bulalım.")

pairs = list(itertools.combinations(range(n), 2))
responses = {}

# Anket Soruları
for (i, j) in pairs:
    st.markdown(f"**{aspects_tr[aspect_keys[i]]}** ⇄ **{aspects_tr[aspect_keys[j]]}**")
    choice = st.radio(key=f"pair_{i}_{j}", label="", options=scale_labels, index=8, horizontal=True)
    responses[(i, j)] = idx_to_saaty(scale_labels.index(choice))

# Analiz Butonu
if st.button("Analizi Gerçekleştir"):
    # AHP Matrisi Oluşturma
    A = np.ones((n, n))
    for (i, j), val in responses.items():
        A[i, j] = val
        A[j, i] = 1.0 / val
    
    weights, CR, lambda_max, CI = ahp_weights(A)

    st.divider()
    st.subheader("1. Kriter Ağırlık Analizi")
    res_df = pd.DataFrame({
        "Değerlendirme Kriteri": [aspects_tr[k] for k in aspect_keys], 
        "Ağırlık Puanı": np.round(weights, 4)
    }).sort_values("Ağırlık Puanı", ascending=False)
    st.table(res_df)
    st.write(f"📊 **Tutarlılık Oranı (CR):** {CR:.4f}")

    # Tutarlılık Kontrolü ve TOPSIS
    if CR <= 0.10:
        st.success("✅ Veriler tutarlı. TOPSIS sıralaması yapılıyor...")
        
        topsis_puanlari = run_topsis(raw_matrix, weights)
        
        ranking = pd.DataFrame({
            "Yapay Zekâ Modeli": models,
            "TOPSIS Skoru": np.round(topsis_puanlari, 4)
        }).sort_values("TOPSIS Skoru", ascending=False)
        
        en_iyi = ranking.iloc[0]["Yapay Zekâ Modeli"]
        
        st.divider()
        st.header(f"🏆 Sizin İçin En Uygun Model: {en_iyi}")
        st.bar_chart(ranking.set_index("Yapay Zekâ Modeli"))
        st.dataframe(ranking, hide_index=True)

        # Sonuçları Kaydet
        if not os.path.exists(RAW_CSV):
            pd.DataFrame(columns=["timestamp", "CR", "Önerilen_Model"]).to_csv(RAW_CSV, index=False)
        
        new_entry = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "CR": round(CR, 4), "Önerilen_Model": en_iyi}
        pd.DataFrame([new_entry]).to_csv(RAW_CSV, mode="a", header=False, index=False)
    else:
        st.error("⚠️ **Tutarsız Yanıtlar!**")
        st.warning("Tutarlılık oranınız (CR) 0.10 limitinin üzerinde çıktı. Lütfen seçimlerinizi daha mantıklı bir dengede tekrar yapın.")
