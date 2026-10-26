
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. KONFIGURASI APLIKASI
# ==========================================

st.set_page_config(
    page_title="Klasifikasi Bunga Iris",
    page_icon="🌸",
    layout="wide"
)


# ==========================================
# 2. MEMUAT DATASET IRIS
# ==========================================

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

nama_bunga = iris.target_names

df = X.copy()

df["species"] = [
    nama_bunga[i] for i in y
]


# ==========================================
# 3. MEMBAGI DATA TRAINING DAN TESTING
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. MEMBUAT MODEL MACHINE LEARNING
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# ==========================================
# 5. EVALUASI MODEL
# ==========================================

y_pred = model.predict(X_test)

akurasi = accuracy_score(
    y_test,
    y_pred
)


# ==========================================
# 6. JUDUL APLIKASI
# ==========================================

st.title("🌸 Aplikasi Klasifikasi Bunga Iris")

st.write(
    """
    Aplikasi Machine Learning untuk mengklasifikasikan
    jenis bunga Iris berdasarkan ukuran sepal dan petal.
    """
)

st.divider()


# ==========================================
# 7. INFORMASI MODEL
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Jumlah Dataset",
        len(df)
    )

with col2:
    st.metric(
        "Jumlah Jenis Bunga",
        len(nama_bunga)
    )

with col3:
    st.metric(
        "Akurasi Model",
        f"{akurasi * 100:.2f}%"
    )


st.divider()


# ==========================================
# 8. SIDEBAR INPUT
# ==========================================

st.sidebar.header("🌱 Input Karakteristik Bunga")

st.sidebar.write(
    "Masukkan ukuran bunga untuk melakukan klasifikasi."
)


sepal_length = st.sidebar.slider(
    "Sepal Length (cm)",
    min_value=4.0,
    max_value=8.0,
    value=5.1,
    step=0.1
)

sepal_width = st.sidebar.slider(
    "Sepal Width (cm)",
    min_value=2.0,
    max_value=4.5,
    value=3.5,
    step=0.1
)

petal_length = st.sidebar.slider(
    "Petal Length (cm)",
    min_value=1.0,
    max_value=7.0,
    value=1.4,
    step=0.1
)

petal_width = st.sidebar.slider(
    "Petal Width (cm)",
    min_value=0.1,
    max_value=2.5,
    value=0.2,
    step=0.1
)


# ==========================================
# 9. DATA INPUT PREDIKSI
# ==========================================

data_input = pd.DataFrame(
    [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]],
    columns=iris.feature_names
)


# ==========================================
# 10. MELAKUKAN PREDIKSI
# ==========================================

prediksi = model.predict(data_input)

probabilitas = model.predict_proba(
    data_input
)

hasil_bunga = nama_bunga[prediksi[0]]


# ==========================================
# 11. MENAMPILKAN HASIL
# ==========================================

st.subheader("🔍 Hasil Klasifikasi")

col1, col2 = st.columns(2)

with col1:

    st.success(
        f"Jenis bunga diprediksi: {hasil_bunga.upper()}"
    )

    st.write(
        "Karakteristik bunga yang dimasukkan:"
    )

    st.dataframe(
        data_input,
        use_container_width=True
    )

with col2:

    st.write("Probabilitas setiap jenis bunga:")

    probabilitas_df = pd.DataFrame({
        "Jenis Bunga": nama_bunga,
        "Probabilitas": probabilitas[0]
    })

    st.bar_chart(
        probabilitas_df.set_index("Jenis Bunga")
    )


# ==========================================
# 12. VISUALISASI DATASET
# ==========================================

st.divider()

st.subheader("📊 Visualisasi Dataset Bunga Iris")

fig, ax = plt.subplots(figsize=(8, 5))

for jenis in nama_bunga:

    data_jenis = df[
        df["species"] == jenis
    ]

    ax.scatter(
        data_jenis["petal length (cm)"],
        data_jenis["petal width (cm)"],
        label=jenis
    )

ax.set_xlabel("Petal Length (cm)")
ax.set_ylabel("Petal Width (cm)")

ax.set_title(
    "Hubungan Panjang dan Lebar Petal"
)

ax.legend()

st.pyplot(fig)


# ==========================================
# 13. MENAMPILKAN DATASET
# ==========================================

with st.expander("📋 Lihat Dataset Bunga Iris"):

    st.dataframe(
        df,
        use_container_width=True
    )


# ==========================================
# 14. INFORMASI MODEL
# ==========================================

with st.expander("🤖 Informasi Model Machine Learning"):

    st.write(
        """
        Model yang digunakan adalah Random Forest Classifier.

        Dataset dibagi menjadi:
        - 80% data training
        - 20% data testing

        Model menggunakan 100 decision trees.

        Evaluasi model menggunakan accuracy score.
        """
    )

    st.write("Classification Report:")

    report = classification_report(
        y_test,
        y_pred,
        target_names=nama_bunga,
        output_dict=True
    )

    st.dataframe(
        pd.DataFrame(report).transpose(),
        use_container_width=True
    )


# ==========================================
# 15. FOOTER
# ==========================================

st.divider()

st.caption(
    "Aplikasi Klasifikasi Bunga Iris | "
    "Python + Streamlit + Scikit-learn"
)
