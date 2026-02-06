# 📊 Model Forecasting Permintaan Retail

**Menggunakan Linear Regression dan LSTM untuk Optimasi Manajemen Persediaan**

Aplikasi web untuk forecasting permintaan retail menggunakan Linear Regression dan LSTM (Long Short-Term Memory). Model ini dirancang untuk membantu optimasi manajemen persediaan dengan memberikan prediksi akurat tentang permintaan produk di masa depan.

## 🎯 Tujuan

- Mengurangi biaya penyimpanan (holding cost)
- Mencegah stockout (kehabisan stok)
- Optimasi tingkat persediaan (inventory level)
- Meningkatkan efisiensi operasional retail

## 🚀 Fitur

- **Visualisasi Data**: Analisis data historis permintaan retail dengan grafik interaktif
- **Evaluasi Model**: Analisis performa kedua model forecasting (Linear Regression & LSTM)
- **Forecast**: Prediksi permintaan 30 hari ke depan untuk perencanaan persediaan
- **Perbandingan Model**: Perbandingan komprehensif akurasi kedua model secara visual

## 📋 Prerequisites

- Python 3.8 atau lebih baru
- File `sales.csv` dengan struktur data retail:
  - `date`: Kolom tanggal (format: YYYY-MM-DD)
  - Kolom numerik untuk permintaan/penjualan (misalnya: `sum_total`, `quantity`, `sales`, dll)

## 🔧 Instalasi

1. Clone atau download repository ini

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Pastikan file `sales.csv` ada di direktori yang sama dengan `app.py`

## 🎯 Cara Menjalankan

Jalankan aplikasi dengan perintah:

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser secara otomatis di `http://localhost:8501`

## 📖 Struktur Aplikasi

### 🏠 Beranda
- Informasi umum tentang aplikasi
- Preview data
- Statistik data

### 📈 Visualisasi Data
- Grafik time series data historis
- Statistik deskriptif

### 🤖 Evaluasi Model
- Metrik evaluasi (MAE, RMSE)
- Grafik prediksi vs aktual untuk kedua model

### 🔮 Forecast
- Prediksi permintaan 30 hari ke depan untuk perencanaan persediaan
- Grafik forecast dengan data historis untuk analisis trend
- Download hasil forecast dalam format CSV untuk integrasi dengan sistem manajemen persediaan

### 📊 Perbandingan Model
- Grafik perbandingan prediksi kedua model
- Tabel perbandingan metrik
- Bar chart perbandingan MAE dan RMSE

## 📝 Catatan Penting

- Model akan di-training setiap kali halaman evaluasi/forecast dibuka
- Untuk performa lebih baik dalam produksi, disarankan untuk menyimpan model yang sudah di-training
- Pastikan data memiliki minimal 60 hari untuk training LSTM (window size = 60)
- Data yang lebih banyak akan menghasilkan prediksi yang lebih akurat
- Model LSTM lebih baik untuk menangkap pola kompleks dan non-linear dalam data permintaan retail
- Linear Regression cocok untuk data dengan trend yang relatif stabil dan linear

## 💡 Aplikasi dalam Manajemen Persediaan

Model forecasting ini dapat digunakan untuk:
- **Perencanaan Pembelian**: Menentukan jumlah produk yang perlu dibeli berdasarkan prediksi permintaan
- **Optimasi Stok**: Menjaga tingkat persediaan optimal untuk mengurangi biaya penyimpanan
- **Pencegahan Stockout**: Memprediksi kapan stok akan habis dan merencanakan restocking
- **Analisis Trend**: Memahami pola permintaan musiman atau trend jangka panjang

## 🛠️ Teknologi yang Digunakan

- **Streamlit**: Framework untuk membuat aplikasi web
- **Pandas**: Manipulasi data
- **NumPy**: Komputasi numerik
- **Matplotlib**: Visualisasi data
- **Scikit-learn**: Machine learning (Linear Regression)
- **TensorFlow**: Deep learning (LSTM)

## 📄 License

MIT License
