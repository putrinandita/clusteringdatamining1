import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import numpy as np
import warnings

# Mengabaikan peringatan untuk kebersihan output
warnings.filterwarnings("ignore")

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Analisis Clustering Penjualan Vending Machine",
    layout="wide"
)

st.title("Proyek Data Mining: Clustering Penjualan Vending Machine")
st.subheader("Metode Clustering: MiniBatchKMeans (Ensemble-like Method)")

PKL_FILE = 'vending_machine_sales_clustered.pkl'
n_clusters = 5 # Jumlah klaster yang dipilih sebelumnya

# --- Fungsi untuk Menerapkan Preprocessing (diperlukan untuk Plotting) ---
@st.cache_data
def preprocess_data(df_input):
    """Melakukan preprocessing dan standardisasi data."""
    
    # Pilih fitur numerik yang relevan
    features = ['RPrice', 'RQty', 'MPrice', 'MQty', 'LineTotal', 'TransTotal']
    df_cluster = df_input[features].copy()

    # Penanganan Missing Values (Imputasi dengan nilai rata-rata)
    df_cluster['MPrice'].fillna(df_cluster['MPrice'].mean(), inplace=True)

    # Standardisasi Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)
    
    return X_scaled, df_cluster

# --- Memuat Data dan Plot dari File ---

try:
    # 1. Muat data yang sudah di-cluster dari file PKL
    df_clustered = pd.read_pickle(PKL_FILE)
    st.success(f"Data klaster berhasil dimuat dari file {PKL_FILE}!")

    # 2. Persiapan Data untuk Plotting
    X_scaled, _ = preprocess_data(df_clustered)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Data dengan Label Klaster")
        st.dataframe(df_clustered[['TransDate', 'Location', 'Product', 'LineTotal', 'TransTotal', 'Cluster']].head(10))

    with col2:
        st.subheader("📈 Ringkasan Ukuran Klaster")
        cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
        st.bar_chart(cluster_counts)

    st.markdown("---")

    st.subheader("🖼️ Visualisasi Hasil Clustering")

    # --- Plot 1: Elbow Method (Regenerasi Plot) ---
    
    @st.cache_resource
    def generate_elbow_plot(X_scaled):
        wcss = []
        k_values = range(2, 11)
        
        for k in k_values:
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
            
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(k_values, wcss, marker='o', linestyle='--', color='blue')
        ax.set_title('Elbow Method (MiniBatchKMeans)')
        ax.set_xlabel('Jumlah Klaster (k)')
        ax.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
        ax.set_xticks(k_values)
        ax.grid(True)
        
        return fig

    st.markdown("##### 1. Plot Metode Siku (Elbow Method)")
    st.pyplot(generate_elbow_plot(X_scaled))

    # --- Plot 2: PCA Visualization (Regenerasi Plot) ---
    
    @st.cache_resource
    def generate_pca_plot(X_scaled, cluster_labels, n_clusters):
        pca = PCA(n_components=2, random_state=42)
        principal_components = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        
        # Ambil label klaster dari df_clustered
        df_pca['Cluster'] = cluster_labels

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(df_pca['PC1'], df_pca['PC2'], 
                             c=df_pca['Cluster'], 
                             cmap='Spectral', alpha=0.7)
        
        ax.legend(*scatter.legend_elements(), title="Klaster", loc="upper right")
        ax.set_title(f'Visualisasi Klaster dengan PCA ({n_clusters} Klaster)')
        ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
        ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
        ax.grid(True)
        
        return fig

    st.markdown("##### 2. Visualisasi Klaster dengan PCA")
    st.pyplot(generate_pca_plot(X_scaled, df_clustered['Cluster'], n_clusters))

    
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
