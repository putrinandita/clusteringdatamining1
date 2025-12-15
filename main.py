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
features = ['RPrice', 'RQty', 'MPrice', 'MQty', 'LineTotal', 'TransTotal']

# --- Fungsi untuk Menerapkan Preprocessing dan Melatih Model ---
@st.cache_resource
def train_and_get_tools(df_input):
    """Melakukan preprocessing, standardisasi, dan melatih model K-Means."""
    
    # 1. Persiapan Data
    df_cluster = df_input[features].copy()
    df_cluster['MPrice'].fillna(df_cluster['MPrice'].mean(), inplace=True)
    
    # 2. Standardisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)
    
    # 3. Pelatihan Model MiniBatchKMeans (Final Model)
    kmeans_final = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans_final.fit(X_scaled)
    
    # 4. Melabeli data (jika belum ada)
    if 'Cluster' not in df_input.columns:
        df_input['Cluster'] = kmeans_final.labels_
        
    return X_scaled, df_cluster, scaler, kmeans_final, df_input

# --- Memuat Data dan Plot dari File ---

try:
    # 1. Muat data yang sudah di-cluster dari file PKL
    df_clustered = pd.read_pickle(PKL_FILE)
    st.success(f"Data klaster berhasil dimuat dari file {PKL_FILE}!")

    # 2. Latih kembali model dan dapatkan scaler/model untuk prediksi manual
    # Note: Kita perlu melatihnya kembali untuk mendapatkan objek scaler dan model
    X_scaled, df_cluster_features, scaler_model, kmeans_model, df_clustered = train_and_get_tools(df_clustered.copy())

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

    st.markdown("---")
    
    # ==============================================================================
    # START: Input Data Manual untuk Prediksi Klaster
    # ==============================================================================
    
    st.header("🧪 Simulasi Transaksi Baru: Prediksi Klaster")
    st.write("Masukkan nilai transaksi baru untuk melihat klaster mana yang paling cocok (Prediksi Perilaku).")

    with st.form("input_form"):
        col_input_1, col_input_2, col_input_3 = st.columns(3)
        
        # Kolom 1
        r_price = col_input_1.number_input("Harga Eceran (RPrice)", min_value=0.01, value=2.50, step=0.01)
        r_qty = col_input_1.number_input("Kuantitas Eceran (RQty)", min_value=1, value=1, step=1)
        
        # Kolom 2
        m_price = col_input_2.number_input("Harga Mesin (MPrice)", min_value=0.01, value=2.50, step=0.01)
        m_qty = col_input_2.number_input("Kuantitas Mesin (MQty)", min_value=1, value=1, step=1)
        
        # Kolom 3
        line_total = col_input_3.number_input("Total Baris (LineTotal)", min_value=0.01, value=2.50, step=0.01)
        trans_total = col_input_3.number_input("Total Transaksi (TransTotal)", min_value=0.01, value=2.50, step=0.01)
        
        submitted = st.form_submit_button("Klasifikasikan Transaksi!")

    if submitted:
        # 1. Buat DataFrame dari input pengguna
        new_data = pd.DataFrame([[r_price, r_qty, m_price, m_qty, line_total, trans_total]], columns=features)
        
        # 2. Skalakan data baru menggunakan Scaler yang sudah dilatih
        # Penting: Hanya transform, tidak fit lagi!
        new_data_scaled = scaler_model.transform(new_data)
        
        # 3. Prediksi klaster menggunakan model MiniBatchKMeans
        predicted_cluster = kmeans_model.predict(new_data_scaled)[0]
        
        # 4. Tampilkan Hasil
        st.subheader("✅ Hasil Klasifikasi")
        st.success(f"Transaksi ini paling cocok masuk ke **Klaster {predicted_cluster}**!")
        st.info(f"""
        **Interpretasi Klaster:** Klaster {predicted_cluster} mewakili kelompok transaksi dengan rata-rata nilai total transaksi:
        - **Klaster 2:** Transaksi Sangat Murah (Rata-rata $1.13)
        - **Klaster 0:** Transaksi Murah Standar (Rata-rata $1.78)
        - **Klaster 3:** Transaksi Nilai Menengah (Rata-rata $3.10)
        - **Klaster 1:** Transaksi Nilai Tinggi/Volume Menengah (Rata-rata $4.01)
        - **Klaster 4:** Transaksi Premium/Nilai Tertinggi (Rata-rata $4.47)
        """)
    
    # ==============================================================================
    # END: Input Data Manual
    # ==============================================================================

    st.markdown("""
<br>
<div style="padding: 10px; border-left: 5px solid #ff4b4b; background-color: #f0f2f6;">
    <p style="font-size: 1.1em;"><b>Penjelasan Metode MiniBatchKMeans:</b></p>
    <p>Kami menggunakan <b>MiniBatchKMeans</b> sebagai metode <i>clustering</i>. Metode ini bekerja dengan mengambil sebagian kecil (mini-batch) data pada setiap iterasi. Hal ini membuatnya jauh lebih cepat pada <i>dataset</i> besar dibandingkan K-Means tradisional. Sifat stokastik dan penggunaan sub-sampel data pada dasarnya memberikan perspektif yang berbeda pada model pada setiap langkah, sehingga dapat dianggap sebagai teknik yang lebih canggih (ensemble-like) dan efisien.</p>
</div>
""", unsafe_allow_html=True)

except FileNotFoundError as e:
    st.error(f"""
        File data utama tidak ditemukan! ({e})
        Pastikan Anda telah mengunggah file **'{PKL_FILE}'** ke direktori kerja Anda atau GitHub.
    """)
    st.info("Fitur yang digunakan untuk clustering adalah: RPrice, RQty, MPrice, MQty, LineTotal, TransTotal.")
