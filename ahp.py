import os
import itertools
import numpy as np
import pandas as pd
import streamlit as st

# ===================== DOSYA YAPISI =====================
BASE = os.path.join(os.path.expanduser("~"), "Desktop", "TEZ_KODLAR_SON")
os.makedirs(BASE, exist_ok=True)

# ===================== KRİTERLER =====================
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

# ===================== RAW SKOR MATRİSİ =====================
raw_matrix = np.array([
    [0.534583, 0.430275, 0.847357, 0.665037, 0.488741, 0.344586, 0.835551, 0.361533],
    [0.606707, 0.643581, 0.829183, 0.659910, 0.600000, 0.472727, 0.722222, 0.361533],
    [0.697545, 0.629730, 0.910492, 0.832819, 0.659259, 0.590580, 0.815493, 0.805134],
    [0.424970, 0.376968, 0.798503, 0.642006, 0.579021, 0.347259, 0.691142, 1.000000],
    [0.700422, 0.548165, 0.914431, 0.852204, 0.663430, 0.437500, 0.760081, 0.000000],
])

# ===================== SAATY ÖLÇEĞİ =====================
scale_labels = [
    "L9", "L8", "L7", "L6", "L5", "L4", "L3", "L2",
    "EQ",
    "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"
]

def get_saaty_value(label: str) -> float:
    if label == "EQ":
        return 1.0
    if label.startswith("L"):
        return float(int(label[1:]))
    if label.startswith("R"):
        return 1.0 / float(int(label[1:]))
    raise ValueError("Geçersiz ölçek")

# ===================== AHP =====================
def calculate_ahp(A: np.ndarray):
    eigvals, eigvecs = np.linalg.eig(A)
    idx = int(np.argmax(eigvals.real))
    w = np.abs(eigvecs[:, idx].real)
    w = w / w.sum()

    lambda_max = float(eigvals[idx].real)
    CI = (lambda_max - n) / (n - 1)
    RI = {3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41}.get(n, 1.45)
    CR = CI / RI if RI > 0 else 0.0
    return w, CR

# ===================== STREAMLIT =====================
st.set_page_config(page_title="Karar Destek Sistemi", layout="wide")
st.title("Size En Uygun Yapay Zekâ Asistanı Önerisi")

st.markdown("""
Bu anket, kullanıcıların yapay zekâ uygulamalarını değerlendirirken
hangi özelliklere daha fazla önem verdiğini belirlemek amacıyla hazırlanmıştır.

Anket kapsamında, her adımda iki farklı özellik karşılaştırılacaktır
(örneğin *Eğitim/Öğrenme* ile *Fiyat/Maliyet*).
Orta alanda yer alan seçim çubuğunu kullanarak, sizin için
hangi özelliğin daha önemli olduğunu belirtmeniz beklenmektedir.

Seçim ölçeği aşağıdaki şekilde yorumlanmalıdır:

- **EQ**: Her iki özellik eşit derecede önemlidir  
- **L2 – L9**: Soldaki özellik daha önemlidir  
  *(değer arttıkça önem farkı büyür)*  
- **R2 – R9**: Sağdaki özellik daha önemlidir  
  *(değer arttıkça önem farkı büyür)*  

Tüm karşılaştırmalar tamamlandıktan sonra **HESAPLAMAYI BAŞLAT**
butonuna basıldığında, sistem belirlenen önceliklere göre
uygulamaları puanlayacak ve en uygun seçeneği önerecektir.
""")

st.divider()

# ===================== PAIRWISE =====================
pairs = list(itertools.combinations(range(n), 2))
responses = {}

st.subheader("1) Önceliklerinizi Seçin (Kıyaslama)")

for (i, j) in pairs:
    l_name = aspects_tr[aspect_keys[i]]
    r_name = aspects_tr[aspect_keys[j]]

    col1, col2, col3 = st.columns([2, 5, 2])
    col1.markdown(f"**{l_name}**")
    col3.markdown(f"**{r_name}**")

    val = col2.select_slider(
        "Karşılaştırma",
        options=scale_labels,
        value="EQ",
        key=f"p_{i}_{j}",
        label_visibility="collapsed"
    )
    responses[(i, j)] = get_saaty_value(val)

# ===================== HESAPLAMA =====================
st.divider()
if st.button("HESAPLAMAYI BAŞLAT", use_container_width=True, type="primary"):
    A = np.ones((n, n))
    for (i, j), v in responses.items():
        A[i, j] = v
        A[j, i] = 1 / v

    weights, CR = calculate_ahp(A)

    if CR > 0.10:
        st.error(f"Seçimleriniz birbiriyle biraz çelişiyor (Tutarlılık Oranı CR = {CR:.3f}). "
                 f"Lütfen birkaç karşılaştırmayı gözden geçirip tekrar deneyin.")
        st.stop()

    # ===================== SAW =====================
    saw_scores = raw_matrix @ weights

    ranking = pd.DataFrame({
        "Model": models,
        "Skor": np.round(saw_scores, 4)
    }).sort_values("Skor", ascending=False).reset_index(drop=True)

    # ===================== SONUÇ EKRANI =====================
    st.subheader("2) Sonuçlar")

    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Kriter Ağırlıkları (Sizin Öncelikleriniz)")
        weights_df = pd.DataFrame({
            "Kriter": [aspects_tr[k] for k in aspect_keys],
            "Ağırlık": np.round(weights, 4)
        }).sort_values("Ağırlık", ascending=False).reset_index(drop=True)
        st.table(weights_df)
        st.caption(f"Tutarlılık Oranı (CR): {CR:.4f} (0.10 ve altı iyi kabul edilir)")

    with right:
        st.markdown(f"### Önerilen Model: **{ranking.iloc[0]['Model']}**")
        st.table(ranking)

        st.markdown("### Model Skor Grafiği")
        st.bar_chart(ranking.set_index("Model")[["Skor"]])
