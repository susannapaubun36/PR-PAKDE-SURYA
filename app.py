
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==================================================
# 1. KONFIGURASI APLIKASI
# ==================================================

st.set_page_config(
    page_title="Klasifikasi Bunga Iris",
    page_icon="🌸",
    layout="wide"
)


# ==================================================
# 2. MEMUAT DATASET DAN MELATIH MODEL
# ==================================================

@st.cache_resource
def siapkan_model():
    iris = load_iris()

    X = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    y = iris.target
    nama_bunga = list(iris.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    akurasi = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        target_names=nama_bunga,
        output_dict=True,
        zero_division=0
    )

    df = X.copy()
    df["species"] = [nama_bunga[i] for i in y]

    return (
        iris,
        df,
        model,
        nama_bunga,
        akurasi,
        X_test,
        y_test,
        y_pred,
        report
    )


# ==================================================
# 3. MEMUAT MODEL DENGAN PENANGANAN ERROR
# ==================================================

try:
    (
        iris,
        df,
        model,
        nama_bunga,
        akurasi,
        X_test,
        y_test,
        y_pred,
        report
    ) = siapkan_model()

except Exception as error:
    st.error("Aplikasi gagal memuat dataset atau model.")
    st.exception(error)
    st.stop()


# ==================================================
# 4. JUDUL APLIKASI
# ==================================================

st.title("🌸 Aplikasi Klasifikasi Bunga Iris")

st.write(
    """
    Aplikasi Machine Learning untuk memprediksi jenis bunga
    Iris berdasarkan panjang dan lebar sepal serta petal.
    """
)

st.divider()


# ==================================================
# 5. RINGKASAN DATASET DAN MODEL
# ==================================================

st.subheader("📌 Ringkasan Dataset dan Model")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Jumlah Dataset**")
    st.title("150")
    st.caption("Data bunga Iris")

with col2:
    st.write("**Jumlah Jenis Bunga**")
    st.title("3")
    st.caption("Setosa, Versicolor, Virginica")

with col3:
    st.write("**Akurasi Model**")
    st.title(f"{akurasi * 100:.2f}%")
    st.caption("Pengujian pada data testing")

st.divider()


# ==================================================
# 6. INPUT KARAKTERISTIK BUNGA
# ==================================================

st.sidebar.header("🌱 Input Karakteristik Bunga")

st.sidebar.write(
    "Atur nilai berikut untuk memprediksi jenis bunga."
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


# ==================================================
# 7. MELAKUKAN PREDIKSI
# ==================================================

data_input = pd.DataFrame(
    [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]],
    columns=iris.feature_names
)

try:
    prediksi = int(model.predict(data_input)[0])

    probabilitas = model.predict_proba(data_input)[0]

    hasil_bunga = nama_bunga[prediksi]

except Exception as error:
    st.error("Prediksi gagal dilakukan.")
    st.exception(error)
    st.stop()


# ==================================================
# 8. MENAMPILKAN HASIL KLASIFIKASI
# ==================================================

st.subheader("🔍 Hasil Klasifikasi")

col1, col2 = st.columns(2)

with col1:
    st.success(
        f"Jenis bunga yang diprediksi: {hasil_bunga.upper()}"
    )

    st.write("**Data karakteristik bunga**")

    st.dataframe(
        data_input,
        use_container_width=True,
        hide_index=True
    )

with col2:
    st.write("**Probabilitas setiap jenis bunga**")

    probabilitas_df = pd.DataFrame({
        "Jenis Bunga": nama_bunga,
        "Probabilitas (%)": probabilitas * 100
    })

    st.bar_chart(
        probabilitas_df.set_index("Jenis Bunga"),
        y="Probabilitas (%)"
    )

    st.caption(
        "Probabilitas merupakan estimasi dari model Random Forest, "
        "bukan jaminan kepastian jenis bunga."
    )


# ==================================================
# 9. VISUALISASI DATASET
# ==================================================

st.divider()

st.subheader("📊 Visualisasi Dataset Bunga Iris")

fig, ax = plt.subplots(figsize=(8, 5))

for jenis in nama_bunga:
    data_jenis = df[df["species"] == jenis]

    ax.scatter(
        data_jenis["petal length (cm)"],
        data_jenis["petal width (cm)"],
        label=jenis,
        alpha=0.75
    )

ax.set_xlabel("Panjang Petal (cm)")
ax.set_ylabel("Lebar Petal (cm)")
ax.set_title("Hubungan Panjang dan Lebar Petal")
ax.legend(title="Jenis Bunga")
ax.grid(True, alpha=0.2)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


# ==================================================
# 10. MENAMPILKAN DATASET
# ==================================================

with st.expander("📋 Lihat Dataset Bunga Iris"):
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.write("**Jumlah data setiap jenis bunga**")

    jumlah_bunga = (
        df["species"]
        .value_counts()
        .rename_axis("Jenis Bunga")
        .reset_index(name="Jumlah")
    )

    st.dataframe(
        jumlah_bunga,
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# 11. INFORMASI DAN EVALUASI MODEL
# ==================================================

with st.expander("🤖 Informasi Model Machine Learning"):
    st.write("**Algoritma:** Random Forest Classifier")
    st.write("**Jumlah decision trees:** 100")
    st.write("**Data training:** 80%")
    st.write("**Data testing:** 20%")
    st.write("**Random state:** 42")

    st.write("**Classification Report**")

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )


# ==================================================
# 12. FOOTER
# ==================================================

st.divider()

st.caption(
    "Aplikasi Klasifikasi Bunga Iris | "
    "Python | Streamlit | Scikit-learn"
)
