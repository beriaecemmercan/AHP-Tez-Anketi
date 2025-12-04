# ahp_streamlit.py
# Türkçe arayüz | İngilizce kriter adlarıyla kayıt | AHP ağırlık + Tutarlılık (CR)

import os
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st

# ---- Yollar ----
BASE = r"C:\Users\beria\Desktop\tez_yorumlar" # Kendi yoluyla güncelleyin
RESP_DIR = os.path.join(BASE, "survey_responses")
os.makedirs(RESP_DIR, exist_ok=True)
RAW_CSV = os.path.join(RESP_DIR, "ahp_raw.csv") # Her katılımcı cevabı buraya eklenecek

# ---------------------------------------------------------------------
# ---- KRİTERLER: NİHAİ 9 ASPECT SETİ (SADECE TÜRKÇE GÖSTERİM) ----
# ---------------------------------------------------------------------

# İngilizce anahtar kelimeler, Türkçe görünen isimlerle eşleştirilmiştir.
aspects_tr = {
    "Price/Cost": "Fiyat/Maliyet",
    "Reliability/Error": "Güvenilirlik/Hata",
    "Code & Development": "Kod & Geliştiricilik",
    "Creativity/Visual": "Yaratıcılık/Görsel",
    "Education/Learning": "Eğitim/Öğrenme",
    "Communication/Interaction": "İletişim/Etkileşim",
    "Interface/Usability": "Arayüz/Kolaylık",
    "Language Support": "Dil Desteği",
    "Accuracy/Consistency": "Doğruluk/Bilgi Tutarlılığı"
}

aspect_keys = list(aspects_tr.keys())
n = len(aspect_keys) # n = 9

# ---- 1–9 çift taraflı ölçek (sadece sayılar görünüyor) ----
scale_labels = ["9", "8", "7", "6", "5", "4", "3", "2", "1",
                "2", "3", "4", "5", "6", "7", "8", "9"]
scale_magnitudes = [9, 8, 7, 6, 5, 4, 3, 2, 1,
                    2, 3, 4, 5, 6, 7, 8, 9]

def idx_to_saaty(idx: int) -> float:
    """0..16 indexini Saaty değerine çevirir."""
    mag = scale_magnitudes[idx]
    if idx < 8:        # sol tarafta: sol kriter daha önemli
        return float(mag)
    elif idx == 8:     # ortadaki 1: eşit önemli
        return 1.0
    else:              # sağ tarafta: sağ kriter daha önemli -> 1/mag
        return 1.0 / float(mag)

# ---- Yardımcı fonksiyonlar (Değişmedi) ----
def build_matrix_from_pairs(pairs_dict):
    A = np.ones((n, n), dtype=float)
    for (i, j), val in pairs_dict.items():
        A[i, j] = val
        A[j, i] = 1.0 / val
    return A

def ahp_weights(A):
    # Ana özdeğer ve özvektör (Principal Eigenvector) hesabı
    eigvals, eigvecs = np.linalg.eig(A)
    idx = np.argmax(eigvals.real)
    w = eigvecs[:, idx].real
    w = np.abs(w)
    w = w / w.sum() # Normalize edilmiş ağırlıklar

    lambda_max = eigvals[idx].real
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0

    # Tutarlılık için Rastgele İndeks (RI) Tablosu (n=9 için 1.45)
    RI_table = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
                6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI = RI_table.get(n, 1.49)
    CR = CI / RI if RI > 0 else 0.0
    return w, CR, lambda_max, CI

def init_raw_csv_if_needed():
    if not os.path.exists(RAW_CSV):
        cols = ["timestamp"]
        for (i, j) in itertools.combinations(range(n), 2):
            a = aspect_keys[i]
            b = aspect_keys[j]
            cols.append(f"{a} vs {b}")
        for a in aspect_keys:
            cols.append(f"weight::{a}")
        cols += ["lambda_max", "CI", "CR"]
        pd.DataFrame(columns=cols).to_csv(RAW_CSV, index=False, encoding="utf-8-sig")

def append_response_row(pairs_dict, weights, lambda_max, CI, CR):
    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    for (i, j), val in pairs_dict.items():
        a = aspect_keys[i]
        b = aspect_keys[j]
        row[f"{a} vs {b}"] = val
    for a, w in zip(aspect_keys, weights):
        row[f"weight::{a}"] = float(w)
    row["lambda_max"] = float(lambda_max)
    row["CI"] = float(CI)
    row["CR"] = float(CR)
    df = pd.DataFrame([row])
    df.to_csv(RAW_CSV, mode="a", header=False, index=False, encoding="utf-8-sig")

# ---- UI ----
st.set_page_config(page_title="AHP Anketi — Üretken Yapay Zekâ", page_icon="🧮", layout="centered")
st.title("AHP Anketi — Üretken Yapay Zekâ Uygulamaları (9 Kriter)")

st.markdown(
    """
Bu anket, üretken yapay zekâ uygulamalarını değerlendirmek için **9 kriterin ağırlıklarını** hesaplamamıza yardımcı olur. Her satırda soldaki ve sağdaki kriteri **1–9 ölçeğinde** karşılaştırmanız beklenmektedir.

- Sol tarafa doğru 9'a yaklaşmak: **soldaki kriter (çok) daha önemli**
- Sağ tarafa doğru 9'a yaklaşmak: **sağdaki kriter (çok) daha önemli**
- Ortadaki **1**: iki kriter **eşit derecede önemli**
"""
)

# İngilizce Kriterler bölümü kaldırıldı, sadece Türkçe sunuluyor.
st.divider()

pairs = list(itertools.combinations(range(n), 2))
responses = {}

for (i, j) in pairs:
    left_eng = aspect_keys[i]
    right_eng = aspect_keys[j]
    left_tr = aspects_tr[left_eng]
    right_tr = aspects_tr[right_eng]

    st.markdown(f"**{left_tr}**  ⇄  **{right_tr}**")
    st.caption("Soldan sağa: soldaki kriter 9 → 1 → 9 sağdaki kriter")

    choice = st.radio(
        key=f"pair_{i}_{j}",
        label="",
        options=scale_labels,
        index=8,  # ortadaki 1 = eşit
        horizontal=True,
    )

    idx = scale_labels.index(choice)
    val = idx_to_saaty(idx)
    responses[(i, j)] = val

    st.write("")

submitted = st.button("Karşılaştırmaları Gönder ve Ağırlıkları Hesapla")

if submitted:
    A = build_matrix_from_pairs(responses)
    weights, CR, lambda_max, CI = ahp_weights(A)

    res_df = pd.DataFrame({
        "Kriter": [aspects_tr[k] for k in aspect_keys], # Sadece Türkçe Kriter
        "Ağırlık": np.round(weights, 4)
    }).sort_values("Ağırlık", ascending=False)

    st.subheader("Ağırlık Sonuçları")
    st.dataframe(res_df, hide_index=True, use_container_width=True)

    st.subheader("Tutarlılık Kontrolü (Consistency)")
    st.write(f"λ_max: **{lambda_max:.4f}**  |  CI: **{CI:.4f}**  |  CR: **{CR:.4f}**")

    if CR <= 0.10:
        st.success("Tutarlılık oranı (CR) kabul edilebilir düzeyde (≤ 0.10).")
    else:
        st.warning("Tutarlılık oranı (CR) yüksek görünüyor (> 0.10). Seçimlerinizi gözden geçirmeniz önerilir.")

    init_raw_csv_if_needed()
    append_response_row(responses, weights, lambda_max, CI, CR)

    st.info(f"Cevabınız kaydedildi: {RAW_CSV}")