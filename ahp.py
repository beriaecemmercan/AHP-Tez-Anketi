import os
import itertools
import numpy as np
import pandas as pd
import streamlit as st

BASE = os.path.join(os.path.expanduser("~"), "Desktop", "TEZ_KODLAR_SON")
RESP_DIR = os.path.join(BASE, "survey_responses")
os.makedirs(RESP_DIR, exist_ok=True)
RAW_CSV = os.path.join(RESP_DIR, "ahp_raw.csv")

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

models = ["CHATGPT", "CLAUDE", "COPILOT", "GEMINI", "GROK"]

raw_matrix = np.array([
    [0.534583, 0.430275, 0.847357, 0.665037, 0.488741, 0.344586, 0.835551, 0.361533],
    [0.606707, 0.643581, 0.829183, 0.659910, 0.600000, 0.472727, 0.722222, 0.361533],
    [0.697545, 0.629730, 0.910492, 0.832819, 0.659259, 0.590580, 0.815493, 0.805134],
    [0.424970, 0.376968, 0.798503, 0.642006, 0.579021, 0.347259, 0.691142, 1.000000],
    [0.700422, 0.548165, 0.914431, 0.852204, 0.663430, 0.437500, 0.760081, 0.000000],
])

scale_labels = ["L9", "L8", "L7", "L6", "L5", "L4", "L3", "L2", "EQ", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"]

def get_saaty_value(label: str) -> float:
    if label == "EQ":
        return 1.0
    if label.startswith("L"):
        return float(int(label[1:]))
    if label.startswith("R"):
        return 1.0 / float(int(label[1:]))
    raise ValueError(f"Invalid scale label: {label}")

def calculate_ahp(A: np.ndarray):
    eigvals, eigvecs = np.linalg.eig(A)
    idx = int(np.argmax(eigvals.real))
    w = np.abs(eigvecs[:, idx].real)
    w = w / w.sum()

    lambda_max = float(eigvals[idx].real)
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}.get(n, 1.45)
    CR = CI / RI if RI > 0 else 0.0
    return w, CR

def topsis(decision_matrix: np.ndarray, weights: np.ndarray, benefit_mask: np.ndarray):
    X = decision_matrix.astype(float)

    denom = np.sqrt((X ** 2).sum(axis=0))
    denom[denom == 0] = 1.0
    R = X / denom

    V = R * weights

    ideal = np.zeros(V.shape[1])
    anti = np.zeros(V.shape[1])
    for j in range(V.shape[1]):
        if benefit_mask[j]:
            ideal[j] = V[:, j].max()
            anti[j]  = V[:, j].min()
        else:
            ideal[j] = V[:, j].min()
            anti[j]  = V[:, j].max()

    d_pos = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    d_neg = np.sqrt(((V - anti) ** 2).sum(axis=1))

    cc = d_neg / (d_pos + d_neg + 1e-12)
    return cc, d_pos, d_neg

st.set_page_config(page_title="Tez Analiz Paneli", layout="wide")
st.title("Yapay Zekâ Uygulamaları için Çok Kriterli Karar Destek Sistemi")

pairs = list(itertools.combinations(range(n), 2))
responses = {}

for (i, j) in pairs:
    l_name = aspects_tr[aspect_keys[i]]
    r_name = aspects_tr[aspect_keys[j]]

    col1, col2, col3 = st.columns([2, 5, 2])
    col1.markdown(f"**{l_name}**")
    col3.markdown(f"**{r_name}**")

    val = col2.select_slider(
        f"{l_name} vs {r_name}",
        options=scale_labels,
        value="EQ",
        key=f"p_{i}_{j}",
        label_visibility="collapsed"
    )
    responses[(i, j)] = get_saaty_value(val)

if st.button("HESAPLAMAYI BAŞLAT", use_container_width=True, type="primary"):
    A = np.ones((n, n))
    for (i, j), value in responses.items():
        A[i, j] = value
        A[j, i] = 1.0 / value

    weights, CR = calculate_ahp(A)

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Kriter Ağırlıkları")
        results = pd.DataFrame({
            "Kriter": [aspects_tr[k] for k in aspect_keys],
            "Puan": np.round(weights, 4)
        }).sort_values("Puan", ascending=False)
        st.table(results)
        st.write(f"Tutarlılık Oranı (CR): {CR:.4f}")

        with st.expander("Pairwise Karar Matrisi (A)"):
            st.dataframe(pd.DataFrame(
                A,
                index=[aspects_tr[k] for k in aspect_keys],
                columns=[aspects_tr[k] for k in aspect_keys]
            ))

    with col_b:
        if CR <= 0.10:
            # Varsayım yapmıyoruz: hepsi benefit (yüksek daha iyi)
            benefit_mask = np.array([True] * n)

            cc, d_pos, d_neg = topsis(raw_matrix, weights, benefit_mask)

            ranking = pd.DataFrame({
                "Model": models,
                "TOPSIS Skoru (CC)": np.round(cc, 4),
                "d+": np.round(d_pos, 4),
                "d-": np.round(d_neg, 4),
            }).sort_values("TOPSIS Skoru (CC)", ascending=False)

            st.subheader(f"Önerilen: {ranking.iloc[0]['Model']}")
            st.table(ranking.reset_index(drop=True))
            st.bar_chart(ranking.set_index("Model")[["TOPSIS Skoru (CC)"]])
        else:
            st.error("Tutarsız Seçimler. Lütfen tercihleri gözden geçirin.")

