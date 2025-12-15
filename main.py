import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression # Diperlukan untuk Regresi
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

# --- Fungsi untuk Melatih Model dan mendapatkan tools ---
@st.cache_resource
def train_and_get_tools(df_input):
    """Melakukan preprocessing, standardisasi, dan melatih model K-Means/Scaler."""
    
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

    # ==============================================================================
    # VISUALISASI CLUSTERING & REGRESI
    # ==============================================================================
    
    st.subheader("🖼️ Perbandingan Visualisasi: Clustering vs Regresi")
    
    viz_col1, viz_col2 = st.columns(2)

    # --- Kolom 1: Plot Clustering (PCA Visualization) ---
    with viz_col1:
        st.markdown("##### 1. Visualisasi Klaster dengan PCA (Clustering)")
        
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
            ax.set_title(f'Hasil Clustering: Pengelompokan Perilaku ({n_clusters} Klaster)')
            ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
            ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
            ax.grid(True)
            return fig

        st.pyplot(generate_pca_plot(X_scaled, df_clustered['Cluster'], n_clusters))

    # --- Kolom 2: Plot Regresi (Simulasi Prediksi Nilai) ---
    with viz_col2:
        st.markdown("##### 2. Simulasi Regresi Linier (Prediksi Nilai)")
        
        @st.cache_resource
        def generate_regression_plot(df):
            # Tentukan X dan Y untuk Regresi Sederhana
            X_reg = df[['LineTotal']] 
            Y_reg = df['TransTotal'] 

            regressor = LinearRegression()
            regressor.fit(X_reg, Y_reg)
            Y_pred = regressor.predict(X_reg)

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(df['LineTotal'], df['TransTotal'], color='gray', label='Data Aktual')
            ax.plot(df['LineTotal'], Y_pred, color='red', linewidth=3, label='Garis Prediksi (Regresi)')
            ax.set_title('Regresi: Prediksi TransTotal dari LineTotal')
            ax.set_xlabel('LineTotal (Fitur)')
            ax.set_ylabel('TransTotal (Target)')
            ax.legend()
            ax.grid(True)
            return fig

        st.pyplot(generate_regression_plot(df_clustered)) # Gunakan df_clustered sebagai sumber data

    # ==============================================================================
    # END: VISUALISASI
    # ==============================================================================
    
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
        new_data_scaled = scaler_model.transform(new_data)
        
        # 3. Prediksi klaster menggunakan model MiniBatchKMeans
        predicted_cluster = kmeans_model.predict(new_data_scaled)[0]
        
        # 4. Tampilkan Hasil
        st.subheader("✅ Hasil Prediksi")
        st.success(f"Transaksi ini paling cocok masuk ke **Klaster {predicted_cluster}**!")
        
        # Tambahkan kembali interpretasi klaster berdasarkan rata-rata nilai (seperti yang telah kita bahas)
        cluster_interpretations = {
            2: "Transaksi Sangat Murah (Rata-rata $1.13) - Volume Tinggi",
            0: "Transaksi Murah Standar (Rata-rata $1.78)",
            3: "Transaksi Nilai Menengah (Rata-rata $3.10)",
            1: "Transaksi Nilai Tinggi/Volume Menengah (Rata-rata $4.01)",
            4: "Transaksi Premium/Nilai Tertinggi (Rata-rata $4.47) - Profit Kunci"
        }
        
        st.info(f"**Interpretasi:** Klaster {predicted_cluster} mewakili {cluster_interpretations.get(predicted_cluster, 'Klaster tidak dikenal')}.")
    
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
