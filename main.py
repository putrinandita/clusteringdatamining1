import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Analisis Clustering Penjualan Vending Machine",
    layout="wide"
)

st.title("Proyek Data Mining: Clustering Penjualan Vending Machine")
st.subheader("Metode Clustering: MiniBatchKMeans (Ensemble-like Method)")

# --- Memuat Data dan Plot dari File ---

try:
    # Muat data yang sudah di-cluster dari file PKL
    # Menggunakan pd.read_pickle untuk memuat DataFrame dari PKL
    df_clustered = pd.read_pickle('vending_machine_sales_clustered.pkl')
    st.success("Data klaster berhasil dimuat dari file PKL!")

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

    # Tampilkan Visualisasi PCA (Asumsi file gambar telah disimpan)
    st.subheader("🖼️ Visualisasi Hasil Clustering")
    
    st.image('cluster_pca_visualization.png', caption='Visualisasi Klaster dengan PCA (k=5) ', use_column_width=True)
    
    st.image('elbow_method.png', caption='Plot Metode Siku (Elbow Method) untuk menentukan k ', use_column_width=True)
    
except FileNotFoundError as e:
    st.error(f"""
        File penting tidak ditemukan! ({e})
        Pastikan Anda telah mengunggah file-file berikut ke GitHub, di folder yang sama:
        1. 'vending_machine_sales_clustered.pkl' (Hasil Clustering)
        2. 'cluster_pca_visualization.png' (Visualisasi Klaster)
        3. 'elbow_method.png' (Plot Elbow Method)
    """)
    st.info("Fitur yang digunakan untuk clustering adalah: RPrice, RQty, MPrice, MQty, LineTotal, TransTotal.")


st.markdown("""
<br>
<div style="padding: 10px; border-left: 5px solid #ff4b4b; background-color: #f0f2f6;">
    <p style="font-size: 1.1em;"><b>Penjelasan Metode MiniBatchKMeans:</b></p>
    <p>Kami menggunakan <b>MiniBatchKMeans</b> sebagai metode <i>clustering</i>. Metode ini bekerja dengan mengambil sebagian kecil (mini-batch) data pada setiap iterasi. Hal ini membuatnya jauh lebih cepat pada <i>dataset</i> besar dibandingkan K-Means tradisional. Sifat stokastik dan penggunaan sub-sampel data pada dasarnya memberikan perspektif yang berbeda pada model pada setiap langkah, sehingga dapat dianggap sebagai teknik yang lebih canggih (ensemble-like) dan efisien.</p>
</div>
""", unsafe_allow_html=True)
