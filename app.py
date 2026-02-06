import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Aplikasi Prediksi Penjualan",
    page_icon="📊",
    layout="wide"
)

# Judul aplikasi
st.title("📊 Model Forecasting Permintaan Retail")
st.markdown("### Menggunakan Linear Regression dan LSTM untuk Optimasi Manajemen Persediaan")
st.markdown("---")

# Sidebar untuk navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.selectbox(
    "Pilih Halaman",
    ["🏠 Beranda", "📈 Visualisasi Data", "🤖 Evaluasi Model", "🔮 Forecast", "📊 Perbandingan Model"]
)

# Input file CSV (opsional)
uploaded_file = st.sidebar.file_uploader(
    "Upload data retail (CSV) – opsional",
    type=["csv"],
    help="Jika tidak mengupload file, aplikasi akan mencoba membaca 'sales.csv' di direktori yang sama dengan app."
)


# Fungsi untuk load dan preprocessing data
@st.cache_data
def _load_default_data(csv_path: str):
    """Load CSV (full atau sample) dan lakukan preprocessing."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Preprocessing
    if "date" in df.columns:
        df = df.set_index("date").resample("D").sum().fillna(0)
    elif df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex):
        df = df.resample("D").sum().fillna(0)
    else:
        date_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
        if date_cols:
            df = df.set_index(date_cols[0]).resample("D").sum().fillna(0)

    return df


def load_data(uploaded_file):
    """
    Jika user upload CSV → pakai file itu.
    Kalau tidak → coba baca 'sales.csv' lokal.
    """
    # Jika user upload file sendiri
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "date" not in df.columns:
                st.error("File yang di-upload harus memiliki kolom 'date'.")
                return None

            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")

            if "date" in df.columns:
                df = df.set_index("date").resample("D").sum().fillna(0)
            elif df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex):
                df = df.resample("D").sum().fillna(0)

            st.success(f"Data berhasil di-load dari file upload: '{uploaded_file.name}'")
            return df
        except Exception as e:
            st.error(f"Gagal membaca file upload: {e}")
            return None

    # Jika tidak ada upload → coba baca file lokal (sales.csv atau sales_sample.csv)
    base_dir = Path(__file__).resolve().parent
    full_path = base_dir / "sales.csv"
    sample_path = base_dir / "sales_sample.csv"

    try:
        if full_path.exists():
            df = _load_default_data(str(full_path))
            st.info("Data dibaca dari file lokal 'sales.csv' (full dataset, di direktori yang sama dengan app.py).")
            return df
        elif sample_path.exists():
            df = _load_default_data(str(sample_path))
            st.warning(
                "Data dibaca dari 'sales_sample.csv' (subset data untuk demo / deployment). "
                "Untuk hasil evaluasi akhir di laporan, gunakan full 'sales.csv' secara lokal."
            )
            return df
        else:
            raise FileNotFoundError("Tidak menemukan 'sales.csv' maupun 'sales_sample.csv'.")
    except Exception as e:
        st.error(
            "Gagal memuat data lokal.\n\n"
            f"Detail: {e}\n\n"
            "Solusi:\n"
            "- Di lokal: simpan 'sales.csv' di folder yang sama dengan 'app.py'.\n"
            "- Untuk GitHub/deploy: buat file kecil 'sales_sample.csv' (misalnya 5–10 ribu baris) dan commit ke repo."
        )
        return None

# Fungsi untuk training model
@st.cache_resource
def train_models(df, target_col='sum_total'):
    # Split data
    train_size = int(len(df) * 0.8)
    train = df[:train_size].copy()
    test = df[train_size:].copy()
    
    # Linear Regression
    train['t'] = np.arange(len(train))
    test['t'] = np.arange(len(train), len(train) + len(test))
    
    if target_col not in train.columns:
        numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
        preferred_cols = ['sum_total', 'quantity', 'price_base']
        target_col = None
        for col in preferred_cols:
            if col in numeric_cols:
                target_col = col
                break
        if target_col is None:
            numeric_cols = [col for col in numeric_cols if col != 'Unnamed: 0']
            target_col = numeric_cols[0] if numeric_cols else train.columns[0]
    
    lr = LinearRegression()
    lr.fit(train[['t']], train[target_col])
    test['lr_pred'] = lr.predict(test[['t']])
    
    # LSTM
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[[target_col]])
    
    window = 60
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i, 0])
        y.append(scaled[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(window, 1)),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mae')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    lstm_pred = model.predict(X_test, verbose=0)
    lstm_pred = scaler.inverse_transform(lstm_pred)
    
    # Evaluasi
    y_true_lr = test[target_col].values
    y_lr = test['lr_pred'].values
    
    y_true_lstm_scaled = y_test.copy()
    y_true_lstm = scaler.inverse_transform(y_true_lstm_scaled.reshape(-1, 1)).flatten()
    
    if lstm_pred.ndim > 1:
        lstm_pred = lstm_pred.flatten()
    
    if len(y_true_lstm) != len(lstm_pred):
        min_len = min(len(y_true_lstm), len(lstm_pred))
        y_true_lstm = y_true_lstm[:min_len]
        lstm_pred = lstm_pred[:min_len]
    
    mae_lr = mean_absolute_error(y_true_lr, y_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_true_lr, y_lr))
    
    mae_lstm = mean_absolute_error(y_true_lstm, lstm_pred)
    rmse_lstm = np.sqrt(mean_squared_error(y_true_lstm, lstm_pred))
    
    return {
        'lr': lr,
        'lstm_model': model,
        'scaler': scaler,
        'train': train,
        'test': test,
        'target_col': target_col,
        'window': window,
        'scaled': scaled,
        'y_true_lr': y_true_lr,
        'y_lr': y_lr,
        'y_true_lstm': y_true_lstm,
        'lstm_pred': lstm_pred,
        'mae_lr': mae_lr,
        'rmse_lr': rmse_lr,
        'mae_lstm': mae_lstm,
        'rmse_lstm': rmse_lstm,
        'df': df
    }

# Load data (menggunakan upload jika ada, kalau tidak pakai sales.csv lokal)
df = load_data(uploaded_file)

if df is not None:
    # Beranda
    if page == "🏠 Beranda":
        st.header("Selamat Datang di Model Forecasting Permintaan Retail")
        st.markdown("""
        Aplikasi ini mengimplementasikan model forecasting permintaan retail menggunakan dua pendekatan machine learning:
        
        - **Linear Regression**: Model regresi linear untuk memprediksi trend permintaan secara sederhana dan cepat
        - **LSTM (Long Short-Term Memory)**: Model deep learning yang mampu menangkap pola temporal kompleks dalam data permintaan retail
        
        ### 🎯 Tujuan:
        Model ini dirancang untuk membantu optimasi manajemen persediaan dengan memberikan prediksi akurat tentang permintaan produk di masa depan, sehingga dapat:
        - Mengurangi biaya penyimpanan (holding cost)
        - Mencegah stockout (kehabisan stok)
        - Optimasi tingkat persediaan (inventory level)
        - Meningkatkan efisiensi operasional retail
        
        ### 📋 Fitur Aplikasi:
        1. **Visualisasi Data**: Analisis data historis permintaan retail
        2. **Evaluasi Model**: Perbandingan performa kedua model forecasting
        3. **Forecast**: Prediksi permintaan 30 hari ke depan untuk perencanaan persediaan
        4. **Perbandingan Model**: Analisis komprehensif perbandingan akurasi kedua model
        """)
        
        st.subheader("📊 Informasi Data")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Data", len(df))
        with col2:
            st.metric("Tanggal Awal", df.index[0].strftime('%Y-%m-%d'))
        with col3:
            st.metric("Tanggal Akhir", df.index[-1].strftime('%Y-%m-%d'))
        with col4:
            st.metric("Kolom", len(df.columns))
        
        st.subheader("Preview Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Statistik Data")
        st.dataframe(df.describe(), use_container_width=True)
    
    # Visualisasi Data
    elif page == "📈 Visualisasi Data":
        st.header("Visualisasi Data Historis")
        
        target_col = st.selectbox("Pilih Kolom untuk Visualisasi", df.select_dtypes(include=[np.number]).columns.tolist())
        
        # Grafik time series
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df[target_col], linewidth=2, color='blue')
        ax.set_title(f'Data Historis: {target_col}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Tanggal', fontsize=12)
        ax.set_ylabel('Nilai', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Statistik
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rata-rata", f"{df[target_col].mean():.2f}")
            st.metric("Standar Deviasi", f"{df[target_col].std():.2f}")
        with col2:
            st.metric("Nilai Maksimum", f"{df[target_col].max():.2f}")
            st.metric("Nilai Minimum", f"{df[target_col].min():.2f}")
    
    # Evaluasi Model
    elif page == "🤖 Evaluasi Model":
        st.header("Evaluasi Model")
        
        with st.spinner("Training model... Mohon tunggu..."):
            results = train_models(df)
        
        st.success("Model berhasil di-training!")
        
        # Metrik
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📉 Linear Regression")
            st.metric("MAE", f"{results['mae_lr']:,.2f}")
            st.metric("RMSE", f"{results['rmse_lr']:,.2f}")
            st.metric("Jumlah Sampel", len(results['y_true_lr']))
        
        with col2:
            st.subheader("🧠 LSTM")
            st.metric("MAE", f"{results['mae_lstm']:,.2f}")
            st.metric("RMSE", f"{results['rmse_lstm']:,.2f}")
            st.metric("Jumlah Sampel", len(results['y_true_lstm']))
        
        # Grafik evaluasi
        st.subheader("Grafik Evaluasi Model")
        
        # Linear Regression
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        test_dates = results['test'].index[:len(results['y_true_lr'])]
        ax1.plot(test_dates, results['y_true_lr'], label='Data Aktual', linewidth=2, color='blue')
        ax1.plot(test_dates, results['y_lr'], label='Prediksi Linear Regression', linewidth=2, color='red', linestyle='--')
        ax1.set_title('Linear Regression: Prediksi vs Aktual', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Tanggal', fontsize=12)
        ax1.set_ylabel('Nilai', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig1)
        
        # LSTM
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        window = results['window']
        train_size = int(len(results['df']) * 0.8)
        lstm_start_idx = train_size + window
        lstm_dates = results['df'].index[lstm_start_idx:lstm_start_idx + len(results['y_true_lstm'])]
        ax2.plot(lstm_dates, results['y_true_lstm'], label='Data Aktual', linewidth=2, color='blue')
        ax2.plot(lstm_dates, results['lstm_pred'], label='Prediksi LSTM', linewidth=2, color='green', linestyle='--')
        ax2.set_title('LSTM: Prediksi vs Aktual', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Tanggal', fontsize=12)
        ax2.set_ylabel('Nilai', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)
    
    # Forecast
    elif page == "🔮 Forecast":
        st.header("Forecast Permintaan 30 Hari Ke Depan")
        st.markdown("Prediksi ini dapat digunakan untuk perencanaan manajemen persediaan dan optimasi stok produk.")
        
        with st.spinner("Training model dan membuat forecast... Mohon tunggu..."):
            results = train_models(df)
        
        # Generate forecast
        test = results['test']
        lr = results['lr']
        model = results['lstm_model']
        scaler = results['scaler']
        window = results['window']
        scaled = results['scaled']
        
        # Linear Regression forecast
        last_t = test['t'].iloc[-1]
        future_lr = []
        for i in range(1, 31):
            future_lr.append(lr.predict([[last_t + i]])[0])
        
        # LSTM forecast
        last_seq = scaled[-window:]
        future_lstm = []
        cur_seq = last_seq.copy()
        
        for _ in range(30):
            pred = model.predict(cur_seq.reshape(1, window, 1), verbose=0)
            future_lstm.append(pred[0][0])
            cur_seq = np.append(cur_seq[1:], pred)
        
        future_lstm = scaler.inverse_transform(np.array(future_lstm).reshape(-1, 1))
        
        # Buat DataFrame forecast
        forecast_df = pd.DataFrame({
            'day': pd.date_range(results['df'].index[-1] + dt.timedelta(days=1), periods=30),
            'lr_forecast': future_lr,
            'lstm_forecast': future_lstm.flatten()
        })
        
        # Grafik forecast
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Data historis 30 hari terakhir
        historical_days = 30
        historical_end = results['df'].index[-1]
        historical_start = results['df'].index[-historical_days] if len(results['df']) >= historical_days else results['df'].index[0]
        historical_data = results['df'].loc[historical_start:historical_end, results['target_col']]
        
        ax.plot(historical_data.index, historical_data.values, 
                label='Data Historis', linewidth=2, color='blue', marker='o', markersize=4)
        ax.plot(forecast_df['day'], forecast_df['lr_forecast'], 
                label='Forecast Linear Regression', linewidth=2, color='red', 
                linestyle='--', marker='s', markersize=4)
        ax.plot(forecast_df['day'], forecast_df['lstm_forecast'], 
                label='Forecast LSTM', linewidth=2, color='green', 
                linestyle='--', marker='^', markersize=4)
        ax.axvline(x=historical_end, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='Mulai Forecast')
        
        ax.set_title('Forecast Permintaan Retail 30 Hari Ke Depan', fontsize=16, fontweight='bold')
        ax.set_xlabel('Tanggal', fontsize=12)
        ax.set_ylabel('Nilai', fontsize=12)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Tabel forecast
        st.subheader("Tabel Forecast")
        st.dataframe(forecast_df, use_container_width=True)
        
        # Statistik forecast
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rata-rata Forecast LR", f"{forecast_df['lr_forecast'].mean():,.2f}")
        with col2:
            st.metric("Rata-rata Forecast LSTM", f"{forecast_df['lstm_forecast'].mean():,.2f}")
        with col3:
            st.metric("Selisih Rata-rata", f"{abs(forecast_df['lr_forecast'].mean() - forecast_df['lstm_forecast'].mean()):,.2f}")
        
        # Download button
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast (CSV)",
            data=csv,
            file_name=f"forecast_{dt.datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Perbandingan Model
    elif page == "📊 Perbandingan Model":
        st.header("Perbandingan Model")
        
        with st.spinner("Training model... Mohon tunggu..."):
            results = train_models(df)
        
        # Grafik perbandingan
        st.subheader("Perbandingan Prediksi pada Data Overlap")
        
        window = results['window']
        overlap_start_in_test = window
        overlap_len = min(len(results['y_true_lstm']), len(results['test']) - overlap_start_in_test)
        
        if overlap_len > 0:
            overlap_dates = results['test'].index[overlap_start_in_test:overlap_start_in_test + overlap_len]
            overlap_actual = results['test'][results['target_col']].values[overlap_start_in_test:overlap_start_in_test + overlap_len]
            overlap_lr = results['test']['lr_pred'].values[overlap_start_in_test:overlap_start_in_test + overlap_len]
            overlap_lstm = results['lstm_pred'][:overlap_len]
            
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(overlap_dates, overlap_actual, label='Data Aktual', linewidth=2, color='blue')
            ax.plot(overlap_dates, overlap_lr, label='Prediksi Linear Regression', linewidth=2, color='red', linestyle='--')
            ax.plot(overlap_dates, overlap_lstm, label='Prediksi LSTM', linewidth=2, color='green', linestyle='--')
            ax.set_title('Perbandingan Linear Regression vs LSTM', fontsize=14, fontweight='bold')
            ax.set_xlabel('Tanggal', fontsize=12)
            ax.set_ylabel('Nilai', fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Perbandingan metrik
            st.subheader("Perbandingan Metrik")
            comparison_df = pd.DataFrame({
                'Model': ['Linear Regression', 'LSTM'],
                'MAE': [results['mae_lr'], results['mae_lstm']],
                'RMSE': [results['rmse_lr'], results['rmse_lstm']],
                'Jumlah Sampel': [len(results['y_true_lr']), len(results['y_true_lstm'])]
            })
            st.dataframe(comparison_df, use_container_width=True)
            
            # Bar chart perbandingan
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            ax1.bar(['Linear Regression', 'LSTM'], [results['mae_lr'], results['mae_lstm']], 
                   color=['red', 'green'], alpha=0.7)
            ax1.set_title('Perbandingan MAE', fontsize=12, fontweight='bold')
            ax1.set_ylabel('MAE', fontsize=11)
            ax1.grid(True, alpha=0.3, axis='y')
            
            ax2.bar(['Linear Regression', 'LSTM'], [results['rmse_lr'], results['rmse_lstm']], 
                   color=['red', 'green'], alpha=0.7)
            ax2.set_title('Perbandingan RMSE', fontsize=12, fontweight='bold')
            ax2.set_ylabel('RMSE', fontsize=11)
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig2)
        else:
            st.warning("Tidak ada data overlap untuk perbandingan")

else:
    st.error("File 'sales.csv' tidak ditemukan. Pastikan file ada di direktori yang sama dengan aplikasi.")
