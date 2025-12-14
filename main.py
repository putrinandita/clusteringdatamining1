import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Analisis Clustering Penjualan Vending Machine",
    layout="wide"
)

st.title("Proyek Data Mining: Clustering Penjualan Vending Machine")
st.subheader("Metode Clustering: MiniBatchKMeans (Ensemble-like Method)")

# --- Nama File Aset ---
PKL_FILE = 'vending_machine_sales_clustered.pkl'
PCA_IMAGE = 'cluster_pca_visualization.png'
ELBOW_IMAGE = 'elbow_method.png'


# --- Memuat Data dan Plot dari File ---

try:
    # 1. Muat data yang sudah di-cluster dari file PKL
    df_clustered = pd.read_pickle(PKL_FILE)
    st.success(f"Data klaster berhasil dimuat dari file {PKL_FILE}!")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Data dengan Label Klaster")
        st.dataframe(df_clustered[['TransDate', 'Location', 'Product', 'LineTotal', 'TransTotal', 'Cluster']].head(10))

    with col2:
        st.subheader("📈 Ringkasan Ukuran Klaster")
        cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
        # Menggunakan bar chart Streamlit yang interaktif
        st.bar_chart(cluster_counts)

    st.markdown("---")

    st.subheader("🖼️ Visualisasi Hasil Clustering")

    # 2. Muat dan Tampilkan Visualisasi Klaster PCA
    try:
        pca_img = Image.open(PCA_IMAGE)
        st.image(pca_img, caption='Visualisasi Klaster dengan PCA (k=5)', use_column_width=True)
    except FileNotFoundError:
        st.error(f"Gagal memuat gambar: {PCA_IMAGE}. Pastikan file ini ada di direktori yang sama.")

    # 3. Muat dan Tampilkan Visualisasi Elbow Method
    try:
        elbow_img = Image.open(ELBOW_IMAGE)
        st.image(elbow_img, caption='Plot Metode Siku (Elbow Method) untuk menentukan k', use_column_width=True)
    except FileNotFoundError:
        st.error(f"Gagal memuat gambar: {ELBOW_IMAGE}. Pastikan file ini ada di direktori yang sama.")
    
except FileNotFoundError as e:
    st.error(f"""
        File data utama tidak ditemukan! ({e})
        Pastikan Anda telah mengunggah file **'{PKL_FILE}'** ke GitHub, di folder yang sama.
    """)
    st.info("Fitur yang digunakan untuk clustering adalah: RPrice, RQty, MPrice, MQty, LineTotal, TransTotal.")


st.markdown("""
<br>
<div style="padding: 10px; border-left: 5px solid #ff4b4b; background-color: #f0f2f6;">
    <p style="font-size: 1.1em;"><b>Penjelasan Metode MiniBatchKMeans:</b></p>
    <p>Kami menggunakan <b>MiniBatchKMeans</b> sebagai metode <i>clustering</i>. Metode ini bekerja dengan mengambil sebagian kecil (mini-batch) data pada setiap iterasi. Hal ini membuatnya jauh lebih cepat pada <i>dataset</i> besar dibandingkan K-Means tradisional. Sifat stokastik dan penggunaan sub-sampel data pada dasarnya memberikan perspektif yang berbeda pada model pada setiap langkah, sehingga dapat dianggap sebagai teknik yang lebih canggih (ensemble-like) dan efisien.</p>
</div>
""", unsafe_allow_html=True)
