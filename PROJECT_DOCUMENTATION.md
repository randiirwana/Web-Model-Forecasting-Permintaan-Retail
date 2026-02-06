# Model Forecasting Permintaan Retail Menggunakan Linear Regression dan LSTM untuk Optimasi Manajemen Persediaan

## 📑 Abstrak

Proyek ini mengimplementasikan model forecasting permintaan retail menggunakan dua pendekatan machine learning: Linear Regression dan LSTM (Long Short-Term Memory). Tujuan utama adalah membantu optimasi manajemen persediaan dengan memberikan prediksi akurat tentang permintaan produk di masa depan, sehingga dapat mengurangi biaya penyimpanan, mencegah stockout, dan meningkatkan efisiensi operasional retail.

## 🎯 Latar Belakang

Manajemen persediaan yang efektif merupakan kunci kesuksesan dalam bisnis retail. Masalah utama yang dihadapi:

1. **Overstocking**: Persediaan berlebih menyebabkan biaya penyimpanan tinggi dan risiko produk kadaluarsa
2. **Understocking**: Persediaan kurang menyebabkan stockout, kehilangan penjualan, dan ketidakpuasan pelanggan
3. **Ketidakpastian Permintaan**: Sulit memprediksi permintaan di masa depan dengan akurat

Forecasting yang akurat dapat membantu mengatasi masalah-masalah tersebut dengan memberikan informasi tentang permintaan di masa depan.

## 🔬 Metodologi

### 1. Linear Regression

**Keunggulan:**
- Sederhana dan mudah diinterpretasikan
- Cepat dalam training dan prediksi
- Cocok untuk data dengan trend linear
- Tidak memerlukan banyak data historis

**Keterbatasan:**
- Tidak dapat menangkap pola non-linear
- Tidak dapat menangkap pola musiman yang kompleks
- Asumsi linearitas mungkin tidak sesuai untuk semua kasus

### 2. LSTM (Long Short-Term Memory)

**Keunggulan:**
- Dapat menangkap pola temporal kompleks
- Mampu mempelajari dependensi jangka panjang
- Cocok untuk data time series dengan pola non-linear
- Dapat menangkap pola musiman dan trend secara bersamaan

**Keterbatasan:**
- Memerlukan lebih banyak data untuk training
- Lebih kompleks dan memerlukan waktu training lebih lama
- Memerlukan tuning hyperparameter yang lebih hati-hati

## 📊 Dataset

Dataset yang digunakan adalah data penjualan retail dengan struktur:
- **Kolom `date`**: Tanggal transaksi
- **Kolom numerik**: 
  - `sum_total`: Total penjualan
  - `quantity`: Jumlah produk
  - `price_base`: Harga dasar
  - Kolom lainnya sesuai kebutuhan

### Preprocessing Data

1. Konversi kolom `date` menjadi datetime
2. Sorting data berdasarkan tanggal
3. Resampling harian (daily) dengan aggregasi sum
4. Handling missing values dengan fillna(0)
5. Normalisasi data untuk LSTM menggunakan MinMaxScaler

## 🏗️ Arsitektur Model

### Linear Regression

```
Input: Time index (t)
Output: Predicted value
Model: y = a*t + b
```

### LSTM

```
Architecture:
- Input Layer: (window_size, 1) = (60, 1)
- LSTM Layer: 64 units, return_sequences=False
- Dense Layer: 1 unit (output)
- Optimizer: Adam
- Loss Function: MAE (Mean Absolute Error)
- Epochs: 50
- Batch Size: 32
```

**Window Size**: 60 hari (2 bulan) - digunakan untuk memprediksi hari berikutnya

## 📈 Evaluasi Model

### Metrik Evaluasi

1. **MAE (Mean Absolute Error)**
   - Mengukur rata-rata kesalahan absolut
   - Semakin kecil semakin baik
   - Formula: MAE = (1/n) * Σ|y_true - y_pred|

2. **RMSE (Root Mean Squared Error)**
   - Mengukur akar rata-rata kuadrat kesalahan
   - Lebih sensitif terhadap outlier
   - Formula: RMSE = √[(1/n) * Σ(y_true - y_pred)²]

### Train-Test Split

- **Training Set**: 80% dari total data
- **Test Set**: 20% dari total data

## 🔮 Forecasting

Model dapat melakukan forecasting untuk:
- **Jangka Pendek**: 30 hari ke depan
- Dapat disesuaikan sesuai kebutuhan bisnis

### Proses Forecasting

1. **Linear Regression**: 
   - Menggunakan time index yang berkelanjutan
   - Prediksi berdasarkan trend linear yang dipelajari

2. **LSTM**:
   - Menggunakan sequence terakhir (60 hari) sebagai input
   - Prediksi iteratif untuk 30 hari ke depan
   - Setiap prediksi menjadi input untuk prediksi berikutnya

## 💼 Aplikasi dalam Manajemen Persediaan

### 1. Perencanaan Pembelian
- Menentukan jumlah produk yang perlu dibeli berdasarkan prediksi permintaan
- Mengoptimalkan waktu pembelian untuk mendapatkan harga terbaik

### 2. Optimasi Stok
- Menjaga tingkat persediaan optimal (safety stock)
- Mengurangi biaya penyimpanan (holding cost)
- Mencegah overstocking dan understocking

### 3. Pencegahan Stockout
- Memprediksi kapan stok akan habis
- Merencanakan restocking sebelum terjadi stockout
- Meningkatkan customer satisfaction

### 4. Analisis Trend
- Memahami pola permintaan musiman
- Mengidentifikasi trend jangka panjang
- Mendukung keputusan strategis

## 🚀 Cara Menggunakan Aplikasi

### Instalasi

```bash
pip install -r requirements.txt
```

### Menjalankan Aplikasi

```bash
streamlit run app.py
```

### Fitur Aplikasi

1. **Beranda**: Informasi umum tentang model dan data
2. **Visualisasi Data**: Analisis data historis
3. **Evaluasi Model**: Perbandingan performa model
4. **Forecast**: Prediksi permintaan 30 hari ke depan
5. **Perbandingan Model**: Analisis komprehensif kedua model

## 📝 Kesimpulan

Model forecasting permintaan retail ini memberikan solusi untuk optimasi manajemen persediaan dengan:

1. **Akurasi**: Menggunakan dua pendekatan berbeda untuk validasi
2. **Fleksibilitas**: Dapat disesuaikan dengan berbagai jenis data retail
3. **Kemudahan Penggunaan**: Interface yang user-friendly
4. **Aplikabilitas**: Langsung dapat digunakan untuk perencanaan persediaan

## 🔮 Pengembangan Selanjutnya

1. **Feature Engineering**: Menambahkan fitur eksternal (musim, promosi, dll)
2. **Hybrid Model**: Menggabungkan Linear Regression dan LSTM
3. **Ensemble Methods**: Menggunakan voting atau averaging dari beberapa model
4. **Real-time Forecasting**: Integrasi dengan sistem real-time
5. **Multi-product Forecasting**: Forecasting untuk beberapa produk sekaligus
6. **Confidence Interval**: Menambahkan interval kepercayaan pada prediksi

## 📚 Referensi

- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
- Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time series analysis: forecasting and control. John Wiley & Sons.
- Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: principles and practice. OTexts.

## 👥 Kontributor

Proyek ini dikembangkan untuk tugas besar pembelajaran mesin.

## 📄 License

MIT License
