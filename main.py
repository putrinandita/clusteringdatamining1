import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import numpy as np
import warnings
import io

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
features = ['RPrice', 'RQty', 'MPrice', 'MQty', 'LineTotal', 'TransTotal']

# --- Fungsi untuk Menerapkan Preprocessing ---
@st.cache_data
def preprocess_data(df_input):
    """Melakukan preprocessing dan standardisasi data."""
    
    # Pilih fitur numerik yang relevan
    df_cluster = df_input[features].copy()

    # Penanganan Missing Values (Imputasi dengan nilai rata-rata)
    # Catatan: Ini harus dilakukan setelah memilih fitur, dan sebelum scaling
    for feature in features:
        df_cluster[feature].fillna(df_cluster[feature].mean(), inplace=True)

    # Standardisasi Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)
    
    return X_scaled, df_cluster

# --- Fungsi untuk Melakukan Clustering Ulang ---
@st.cache_data
def perform_clustering(df_input, X_scaled, n_clusters):
    """Melakukan MiniBatchKMeans dan menambahkan label klaster ke DataFrame."""
    
    # Inisialisasi dan latih MiniBatchKMeans
    kmeans_final = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans_final.fit_predict(X_scaled)
    
    # Tambahkan label klaster ke DataFrame
    df_result = df_input.copy()
    df_result['Cluster'] = cluster_labels
    return df_result, cluster_labels

# --- Fungsi Plotting ---

@st.cache_resource
def generate_elbow_plot(X_scaled):
    """Menghasilkan Plot Metode Siku."""
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

@st.cache_resource
def generate_pca_plot(X_scaled, cluster_labels, n_clusters):
    """Menghasilkan Plot Visualisasi PCA."""
    pca = PCA(n_components=2, random_state=42)
    principal_components = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    
    df_pca['Cluster'] = cluster_labels

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(df_pca['PC1'], df_pca['PC2'], 
                         c=df_pca['Cluster'], 
                         cmap='Spectral', alpha=0.7)
    
    ax.legend(*scatter.legend_elements(), title="Klaster", loc="upper right")
    ax.set_title(f'Visualisasi Klaster dengan PCA ({n_clusters} Klaster)')
    # Perlu memastikan rasio varian adalah array, dan PC1/PC2 sudah didefinisikan sebelumnya
    try:
        ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
        ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
    except IndexError:
        pass # Lewati jika array kosong
        
    ax.grid(True)
    return fig

# --- Logika Utama: Coba Muat PKL, Jika Gagal Minta Upload ---

data_source = None
df_clustered = None

try:
    # 1. Coba Muat data yang sudah di-cluster dari file PKL
    df_clustered = pd.read_pickle(PKL_FILE)
    st.success(f"Data klaster berhasil dimuat dari file {PKL_FILE}!")
    data_source = 'pkl'
    df = df_clustered # Gunakan df_clustered sebagai df dasar
    
except FileNotFoundError:
    st.error(f"""
        File data klaster utama **'{PKL_FILE}'** tidak ditemukan.
        Silakan unggah file data mentah (CSV atau Excel) untuk menjalankan analisis.
    """)
    data_source = 'upload'

# --- Bagian Input Data (Selalu Tampilkan jika PKL tidak ada) ---

if data_source == 'upload' or df_clustered is None:
    uploaded_file = st.file_uploader(
        "Pilih file data transaksi (CSV, XLSX)",
        type=['csv', 'xlsx'],
        key="file_uploader"
    )

    if uploaded_file is not None:
        try:
            # Baca file yang diunggah
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            
            st.success("File berhasil diunggah!")
            
            # --- Sidebar untuk Pengaturan Clustering ---
            st.sidebar.header("⚙️ Pengaturan Clustering")
            n_clusters = st.sidebar.slider(
                "Pilih Jumlah Klaster (k)",
                min_value=2, max_value=10, value=5, step=1
            )
            
            st.sidebar.markdown(f"**Fitur yang Digunakan:** {', '.join(features)}")

            # --- Lakukan Clustering pada Data yang Diunggah ---
            
            with st.spinner('Melakukan Preprocessing dan MiniBatchKMeans...'):
                X_scaled, df_cluster_features = preprocess_data(df)
                df_clustered, cluster_labels = perform_clustering(df, X_scaled, n_clusters)
            
            st.success(f"Clustering berhasil dilakukan dengan k={n_clusters}!")
            data_source = 'uploaded_clustered'
            
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: Pastikan file memiliki kolom: {features}")
            st.exception(e)


# --- Tampilan Hasil (Jika data sudah tersedia/diprocess) ---

if df_clustered is not None:
    
    X_scaled, _ = preprocess_data(df_clustered) # Pastikan X_scaled tersedia

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Data dengan Label Klaster")
        st.dataframe(df_clustered[['TransDate', 'Location', 'Product', 'LineTotal', 'TransTotal', 'Cluster']].head(10))

    with col2:
        st.subheader("📈 Ringkasan Ukuran Klaster (Count)")
        cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
        st.bar_chart(cluster_counts)
        st.write(cluster_counts.to_frame(name='Jumlah Transaksi'))

    st.markdown("---")

    st.subheader("🖼️ Visualisasi Hasil Clustering")

    # --- Plot 1: Elbow Method ---
    st.markdown("##### 1. Plot Metode Siku (Elbow Method)")
    # Elbow plot selalu menampilkan k dari 2 sampai 10, terlepas dari k yang dipilih
    st.pyplot(generate_elbow_plot(X_scaled))

    # --- Plot 2: PCA Visualization ---
    st.markdown(f"##### 2. Visualisasi Klaster dengan PCA (k={n_clusters})")
    st.pyplot(generate_pca_plot(X_scaled, df_clustered['Cluster'], n_clusters))

st.markdown("""
<br>
<div style="padding: 10px; border-left: 5px solid #ff4b4b; background-color: #f0f2f6;">
    <p style="font-size: 1.1em;"><b>Penjelasan Metode MiniBatchKMeans:</b></p>
    <p>Kami menggunakan <b>MiniBatchKMeans</b> sebagai metode <i>clustering</i>. Metode ini bekerja dengan mengambil sebagian kecil (mini-batch) data pada setiap iterasi. Hal ini membuatnya jauh lebih cepat pada <i>dataset</i> besar dibandingkan K-Means tradisional. Sifat stokastik dan penggunaan sub-sampel data pada dasarnya memberikan perspektif yang berbeda pada model pada setiap langkah, sehingga dapat dianggap sebagai teknik yang lebih canggih (ensemble-like) dan efisien.</p>
</div>
""", unsafe_allow_html=True)
