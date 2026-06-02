# AI-1 Model Architect - Achmad Rif'an

Dokumen ini berisi dokumentasi teknis menyeluruh mengenai seluruh pekerjaan **AI-1 (Model Architect)** untuk aplikasi ChatKasir. Dokumentasi ini mencakup perancangan arsitektur jaringan saraf, pipeline pelatihan tingkat lanjut menggunakan kustom loop, evaluasi metrik berbasis token, serta spesifikasi skrip parser untuk kebutuhan handover ke tim API.

## Daftar Isi

- [AI-1 Model Architect - Achmad Rif'an](#ai-1-model-architect---achmad-rifan)
  - [Daftar Isi](#daftar-isi)
  - [Struktur Folder \& File](#struktur-folder--file)
  - [Pola Percakapan Dunia Nyata (Scope Dataset)](#pola-percakapan-dunia-nyata-scope-dataset)
    - [\[1\] Pola Sederhana (Single-Item)](#1-pola-sederhana-single-item)
    - [\[2\] Pola Majemuk / Multi-Item (Direct Processing)](#2-pola-majemuk--multi-item-direct-processing)
    - [\[3\] Penulisan Angka \& Modifikasi (Slang / Noise)](#3-penulisan-angka--modifikasi-slang--noise)
  - [Alur Pemrosesan Lengkap: Dari Input Mentah ke Dashboard](#alur-pemrosesan-lengkap-dari-input-mentah-ke-dashboard)
    - [\[1\] Input dari Aplikasi (WhatsApp Copy-Paste)](#1-input-dari-aplikasi-whatsapp-copy-paste)
    - [\[2\] Preprocessing \& Tokenization](#2-preprocessing--tokenization)
    - [\[3\] Prediksi Model Unified NER](#3-prediksi-model-unified-ner)
    - [\[4\] Postprocessing \& Array Mapping](#4-postprocessing--array-mapping)
    - [\[5\] Database \& Tampilan Dashboard](#5-database--tampilan-dashboard)
  - [Arsitektur Model Final: Unified Transformer-NER](#arsitektur-model-final-unified-transformer-ner)
    - [Kamus Pemetaan Tag (7 Kelas)](#kamus-pemetaan-tag-7-kelas)
  - [Custom Training Loop \& MaskedNERLoss](#custom-training-loop--maskednerloss)
    - [Jantung Komputasi: `MaskedNERLoss`](#jantung-komputasi-maskednerloss)
    - [Logika Alur Siklus Gradient Tape (Per Batch):](#logika-alur-siklus-gradient-tape-per-batch)
  - [Evaluasi Metrik \& Penanganan Error](#evaluasi-metrik--penanganan-error)
    - [Laporan Kemampuan Model](#laporan-kemampuan-model)
    - [Analisis Kesalahan Konteks (*Error Analysis*)](#analisis-kesalahan-konteks-error-analysis)
  - [Panduan Setup \& Eksekusi di Google Colab](#panduan-setup--eksekusi-di-google-colab)
    - [Persiapan Struktur Google Drive](#persiapan-struktur-google-drive)
    - [Konfigurasi Runtime GPU \& Dependency](#konfigurasi-runtime-gpu--dependency)
    - [Alur Eksekusi Notebook Pipeline](#alur-eksekusi-notebook-pipeline)


## Struktur Folder & File


```

ai-model/
├── notebooks/
│   ├── 01_model_architecture.ipynb   # Tokenisasi, Regex Multi-Format, & Pelabelan BIO Dataset
│   ├── 02_training.ipynb             # Proses training dengan Custom Loop & Dynamic Weighting
│   └── 03_evaluation.ipynb           # Evaluasi metrik (Classification Report) & Simulasi End-to-End Parser
├── src/
│   ├── model.py                      # Fungsi arsitektur final (Transformer + BiLSTM + NER Head)
│   └── custom_loss.py                # Fungsi loss kustom tunggal (MaskedNERLoss)
├── assets/
│   ├── data/                         # Aset data biner dan parameter konfigurasi
│   │   ├── dataset_chatkasir.npz     # Hasil pembagian Train/Val/Test
│   │   └── model_config.json         # Parameter konfigurasi global (Vocab, Max Length, Num Tags)
│   ├── models/                       # Folder penyimpanan file biner model terbaik hasil training
│   │   ├── chatkasir_saved_model/    # Folder format SavedModel untuk eksport model
│   │   └── chatkasir_model.keras     # File utama model terbaik format Keras
│   └── tokenizers/                   # Folder penyimpanan aset tokenisasi teks
│       └── tokenizer.json            # File eksport WordPiece Tokenizer (Vocab Size: 10000)
├── logs/                             # Log TensorBoard untuk pemantauan training
├── RESEARCH_NOTES.md                 # Catatan referensi ilmiah
├── requirements.txt                  # Daftar dependensi project standar format pip requirements
└── README.md                         # Dokumentasi utama project

```

## Pola Percakapan Dunia Nyata (Scope Dataset)

Aplikasi ChatKasir dirancang untuk mengurai kekacauan teks pesanan dari percakapan WhatsApp UMKM yang sering kali tidak terstruktur, penuh singkatan (*slang*), dan salah tik (*typo*). Dataset sintetis baru mencakup variasi pola berikut secara *native*:

### [1] Pola Sederhana (Single-Item)
Pesanan tunggal dengan susunan Kuantitas (`QTY`) di depan maupun di belakang Produk (`PROD`).
* *Contoh QTY di depan:* `"mas 2 nasi goreng ya"`
* *Contoh PROD di depan:* `"pesen ayam geprek nya 3 porsi dong"`

### [2] Pola Majemuk / Multi-Item (Direct Processing)
Pembeli memesan lebih dari satu jenis menu sekaligus dalam satu baris chat tanpa pembatas formal. **Model mampu memproses seluruh item ini secara langsung tanpa perlu pemotongan iterasi kalimat dari sisi Backend.**
* *Contoh:* `"bang pesen 3 bakso mercon dan 2 es teh manis [SEP] siap mas 3 bakso mercon 45rb dan 2 es teh manis 10rb jadi total harganya 55rb ya"`

### [3] Penulisan Angka & Modifikasi (Slang / Noise)
* **Kuantitas Huruf:** Mengakomodasi ketikan non-digit seperti `"seporsi"`, `"dua cup"`, `"10 pack"`, `"se-thinwall"`.
* **Variasi Format Harga:** Deteksi mandiri teks harga dari penjual setelah token separator `[SEP]` dengan format beragam mulai dari puluhan ribu hingga jutaan rupiah: `45rb`, `17k`, `350000`, `35jt`, `5 juta`.
* **Modifier (Noise Teks):** Kata pelengkap rasa atau metode penyajian seperti `"level dewa"`, `"bumbu pisah"`, `"makan sini"` secara otomatis diabaikan oleh model dan dilabeli sebagai tag `O`.

---

## Alur Pemrosesan Lengkap: Dari Input Mentah ke Dashboard

Bagian ini merinci bagaimana teks pesanan mentah diproses dari ujung ke ujung hingga berhasil direkam ke dalam sistem dashboard pencatatan.

### [1] Input dari Aplikasi (WhatsApp Copy-Paste)
**Penanggung Jawab:** FS-1 Alfan

Pengguna menyalin teks obrolan mentah ke dalam antarmuka aplikasi.
```text
[28/05, 05:26] Pembeli: order paket ayam bakar madu 10 pack sama es kopi susu gula aren 5 cup
[28/05, 06:01] Kasir: siap paket ayam bakar madu harganya 35ribu dan es kopi susu harganya 18ribu jadi total tagihan semuanya 440ribu

```

### [2] Preprocessing & Tokenization

**Penanggung Jawab:** AI-2 Denny & AI-1 Rifan

* **Universal Regex Wrapper:** Menghapus stempel waktu, nama, dan nomor WhatsApp agar lebih bersih.
* **Normalisasi Teks:** Mengoreksi singkatan kritis, meredam kata sapaan, dan meluruskan nominal singkatan ribuan/jutaan menjadi angka bulat murni sebelum diurai WordPiece Tokenizer (Vocab Size: 10.000).

```text
Hasil Preprocessing: "pesan paket ayam bakar madu 10 bungkus es kopi susu gula aren 5 cup [SEP] siap paket ayam bakar madu harga 350000 es kopi susu gula aren harga 18000 total tagihan katering semua 4400000"

```

### [3] Prediksi Model Unified NER

**Penanggung Jawab:** AI-1 Rifan

Teks yang sudah berbentuk ID token dimasukkan ke dalam model jaringan saraf untuk diprediksi kelas tag BIO-nya (Total 7 Kelas).

```text
TOKEN / KATA    | PREDIKSI TAG
------------------------------
paket           | O
ayam            | B-PROD
bakar           | I-PROD
madu            | I-PROD
10              | B-QTY
bungkus         | I-QTY
es              | B-PROD
kopi            | I-PROD
susu            | I-PROD
gula            | I-PROD
aren            | I-PROD
5               | B-QTY
cup             | I-QTY
[SEP]           | O
35000           | B-PRICE
18000           | B-PRICE
440000          | B-PRICE

```

### [4] Postprocessing & Array Mapping

**Penanggung Jawab:** AI-2 Denny

Output tag BIO dari model diolah menggunakan fungsi parser kustom cerdas (Context-Aware Layer) untuk memisahkan domain pembeli-penjual, membersihkan ejaan kuantitas menjadi integer numerik, serta mengalkulasi subtotal secara dinamis.

```json
[
    {
        "product_name": "Ayam Bakar Madu",
        "quantity": 10,
        "price_satuan": 35000,
        "subtotal": 350000
    },
    {
        "product_name": "Es Kopi Susu Gula Aren",
        "quantity": 5,
        "price_satuan": 18000,
        "subtotal": 90000
    }
]

```

### [5] Database & Tampilan Dashboard

**Penanggung Jawab:** FS-2 Reihan & FS-1 Alfan

Data JSON dikirim ke server backend untuk disimpan ke database PostgreSQL, lalu dirender ke dalam tabel riwayat transaksi kasir secara realtime.

---

## Arsitektur Model Final: Unified Transformer-NER

Model V2 meninggalkan pendekatan arsitektur multi-cabang regresi yang rentan terhadap interferensi tugas (*Task Interference*). Model baru menyatukan seluruh tugas ekstraksi ke dalam satu sistem jaringan saraf NLP terpadu berpola **Named Entity Recognition (NER)**.

```
Input Tokens (128,) 
     │
     ▼
[Embedding Layer (Vocab: 10000, Dim: 128, mask_zero=True)]
     │
     ▼
[Transformer Encoder x2 (Heads: 4, FFN Dim: 256)]  <-- Ekstraksi Konteks Global
     │
     ▼
[Bidirectional LSTM (Units: 64, return_sequences=True)] <-- Pemahaman Urutan Sekuensial
     │
     ▼
[Dense Output Head (Units: 7, Activation: Softmax)] <-- Klasifikasi Tag BIO Per Token

```

### Kamus Pemetaan Tag (7 Kelas)

* `0`: `O` (Kata biasa / Noise pelengkap)
* `1`: `B-PROD` (Awal kata nama menu/produk)
* `2`: `I-PROD` (Lanjutan kata nama menu/produk)
* `3`: `B-QTY` (Awal kata kuantitas/jumlah pesanan)
* `4`: `I-QTY` (Lanjutan kata kuantitas/jumlah pesanan)
* `5`: `B-PRICE` (Awal kata harga satuan/total dari kasir)
* `6`: `I-PRICE` (Lanjutan kata harga satuan/total dari kasir)

---

## Custom Training Loop & MaskedNERLoss

Proses pelatihan model di `02_training.ipynb` dikendalikan secara penuh menggunakan **Custom Training Loop (tf.GradientTape)**. Hal ini dilakukan demi menerapkan teknik optimasi tingkat rendah yang tidak didukung oleh fungsi standar Keras `model.fit`.

### Jantung Komputasi: `MaskedNERLoss`

Untuk menanggulangi masalah ketimpangan kelas (*Class Imbalance*) di mana kata biasa (`O`) mendominasi lebih dari 70% kalimat chat, kita merancang *loss function* kustom berbasis `SparseCategoricalCrossentropy` yang dilengkapi sistem pembobotan hukuman (*Class Weighting*) dan *Padding Masking*:

* **Padding Masking:** Bobot otomatis dikalikan `0` jika token yang dibaca adalah token kosong `[PAD]`. Model tidak akan pernah membuang waktu komputasi untuk menghafal ruang kosong.
* **Sistem Hukuman Bobot Dynamic:**
  * Tag `O` (ID: 0) diberi bobot **1.0** (Hukuman normal).
  * Tag `PROD` (ID: 1, 2) diberi bobot **2.0** (Dihukum 2x lipat lebih keras jika model salah tebak nama menu).
  * Tag `QTY` & `PRICE` (ID: 3, 4, 5, 6) diberi bobot **1.5** (Dihukum 1.5x lipat jika salah deteksi jumlah atau harga).

### Logika Alur Siklus Gradient Tape (Per Batch):

1. **Forward Pass:** Model memproses token `x_batch` dan menghasilkan probabilitas tag `pred_ner`.
2. **Masking:** Skrip menghitung `mask_padding` (mengabaikan token 0) secara langsung.
3. **Backpropagation:** Tape merekam seluruh operasi, menghitung gradien kesalahan terhadap bobot saraf yang dapat dilatih (`model.trainable_weights`).
4. **Optimasi Bobot:** Optimizer Adam memperbarui bobot internal saraf model untuk meminimalkan loss di batch berikutnya.

---

## Evaluasi Metrik & Penanganan Error

### Laporan Kemampuan Model

Evaluasi metrik pada `03_evaluation.ipynb` mengisolasi token padding menggunakan metode masking array 1D. Skor murni mencerminkan kompetensi AI dalam mengukur entitas penting berdasarkan pengujian 10.000 baris data test:

```
LAPORAN EVALUASI PERFORMA DATA TEST (PER ENTITAS):
==================================================
              precision    recall  f1-score   support

      B-PROD     0.9995    1.0000    0.9997     11414
      I-PROD     0.9993    0.9998    0.9996     24972
       B-QTY     0.9828    0.9989    0.9907     11366
       I-QTY     0.9534    0.9984    0.9754      3157
     B-PRICE     0.9979    0.9981    0.9980     12093
     I-PRICE     0.9847    0.9898    0.9872       391

   micro avg     0.9936    0.9992    0.9964     63393
   macro avg     0.9863    0.9975    0.9918     63393
weighted avg     0.9937    0.9992    0.9964     63393
==================================================

```

*Model sukses meraih F1-Score rata-rata global sebesar **99.64%** pada data pengujian murni.*

### Analisis Kesalahan Konteks (*Error Analysis*)

Hasil audit menyeluruh terhadap sisa error minor membuktikan fenomena yang menguntungkan: Model AI terdeteksi lebih pintar daripada label data latih mentahnya (Generalisasi Pintar).

Model berhasil memprediksi kata angka penunjuk jumlah seperti "1" (Kasus 1), "dua bungkus" (Kasus 2), "porsi gede" (Kasus 3), "10" (Kasus 4), dan "porsi kecil" (Kasus 5) secara tepat sebagai entitas kuantitas (B-QTY/I-QTY), namun disalahkan oleh sistem evaluasi karena kunci jawaban di dataset sintetis tidak sengaja terlewat (berlabel O). Ini membuktikan model tidak mengalami hafalan (overfitting) dan memiliki nalar konteks yang sangat matang.

---

## Panduan Setup & Eksekusi di Google Colab

Bagian ini memandu developer atau tim handover untuk mereproduksi proses tokenisasi, training, dan evaluasi menggunakan layanan Google Colab secara runut.

### Persiapan Struktur Google Drive

Sebelum membuka Google Colab, pastikan seluruh file proyek ai-model telah diunggah ke Google Drive dengan struktur struktur folder berikut agar pembacaan otomatis script tidak mengalami FileNotFoundError:

```
Google Drive (My Drive)
└── ChatKasir/
    ├── assets/
    │   ├── data/
    │   ├── models/
    │   └── tokenizers/
    ├── logs/
    ├── 01_model_architecture.ipynb
    ├── 02_training.ipynb
    ├── 03_evaluation.ipynb
    └── requirements.txt
```

### Konfigurasi Runtime GPU & Dependency

1. Buka Google Colab lalu pasang salah satu notebook dari folder notebooks/ pada repository GitHub ini (misalnya 01_model_architecture.ipynb).
2. Aktifkan akselerator GPU T4 pada menu Runtime > Change runtime type > pilih T4 GPU.
3. Jalankan blok kode inisialisasi lingkungan pada notebook cell pertama untuk menghubungkan penyimpanan Drive dan menginstal seluruh pustaka pendukung proyek:

```
from google.colab import drive
import os

# Hubungkan Drive dan langsung pindah ke folder ChatKasir
drive.mount('/content/drive')
%cd /content/drive/MyDrive/ChatKasir

# Cek requirements.txt dan install seluruh library
if os.path.exists('requirements.txt'):
    !pip install -r requirements.txt -q
    print("Semua library terpasang. ChatKasir siap dijalankan")
else:
    print("File 'requirements.txt' tidak ditemukan di folder ChatKasir.")
```

### Alur Eksekusi Notebook Pipeline
Untuk menjalankan atau melatih ulang model secara utuh, eksekusi berkas notebook sesuai dengan urutan logis penanganan data di bawah ini:

1. 01_model_architecture.ipynb
   Tujuan: Membersihkan noise pesan WhatsApp mentah via regex, membangun kosa kata WordPiece Tokenizer berkapasitas 10.000 token unik, memetakan penempatan label token BIO, serta mengekspor biner kompresi dataset dataset_chatkasir.npz.
2. 02_training.ipynb
   Tujuan: Memuat dataset terkompresi, menyusun arsitektur kustom Transformer-BiLSTM, serta mengeksekusi siklus loop pelatihan menggunakan tf.GradientTape bersistem bobot hukuman loss. Output terbaik otomatis terekspor ke dalam bentuk file chatkasir_model.keras dan chatkasir_saved_model.
3. 03_evaluation.ipynb
   Tujuan: Memvalidasi akurasi model akhir pada subset pengujian murni, mencetak tabel klasifikasi performa presisi entitas, serta menjalankan simulasi fungsional fungsi parser teks pesanan mentah menjadi keluaran JSON siap pakai oleh tim backend.