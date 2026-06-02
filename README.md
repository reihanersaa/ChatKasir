# ChatKasir - Asisten Kasir Cerdas Berbasis AI

![ChatKasir Banner](docs/bannerchatkasir.png)

**ChatKasir** adalah sistem aplikasi pencatatan transaksi kasir cerdas yang dirancang khusus untuk UMKM Indonesia. Aplikasi ini memanfaatkan **Artificial Intelligence (AI) - Named Entity Recognition (NER)** untuk membaca, mengekstrak, dan menghitung pesanan secara otomatis langsung dari teks obrolan (seperti dari WhatsApp).

Selain itu, ChatKasir dilengkapi dengan asisten chatbot **TanyaAI** (berbasis Google Gemini) untuk mengedukasi pemilik UMKM seputar bisnis, serta **Dashboard Analitik** untuk memantau performa penjualan.

---

## Tautan Aplikasi Resmi (Live Deployment)

Anda dapat mengakses dan mencoba ekosistem ChatKasir yang sudah berjalan secara langsung di cloud melalui tautan berikut:

- **Aplikasi Utama (Frontend - Vercel):** [https://chatkasir.vercel.app]
- **Core Service (Backend API - Vercel):** [https://chat-kasir-backend.vercel.app]
- **Asisten Cerdas (AI Inference API - Hugging Face Spaces):** [https://achmadrifan-chatkasir.hf.space]
- **Dashboard Analitik & Bisnis (Streamlit Cloud):** [https://dashboard-chatkasir.streamlit.app]

---

## Struktur Proyek & Panduan Navigasi Bidang

Proyek ini dibangun menggunakan arsitektur _Decoupled Microservices_. Jika Anda ingin mempelajari alur kerja spesifik dari masing-masing sub-tim, gunakan tabel panduan berikut untuk menuju ke file dokumentasi teknis yang sesuai:

| Jika Anda ingin mempelajari/mengembangkan...                                     | Maka buka Direktori  | Dokumentasi Lengkap                              |
| :------------------------------------------------------------------------------- | :------------------- | :----------------------------------------------- |
| Tampilan UI, State Management React, Styling Tailwind & DaisyUI                  | **`/frontend`**      | [Baca README Frontend](./frontend/README.md)     |
| API Gateway, Autentikasi Supabase, Endpoint Transaksi, & Logika Bisnis           | **`/backend`**       | [Baca README Backend](./backend/README.md)       |
| Pipeline Inferensi Model, Preprocessing Teks Chat, & FastAPI Service             | **`/api-inference`** | [Baca README AI API](./api-inference/README.md)  |
| Eksperimen Arsitektur Jaringan Saraf NER, Kustom Training Loop, & Model `.keras` | **`/ai-model`**      | [Baca README AI Architect](./ai-model/README.md) |
| Visualisasi Interaktif Plotly, Pertanyaan Bisnis, & Dashboard Monitoring         | **`/dashboard`**     | [Baca README Dashboard](./dashboard/README.md)   |
| Analisis Data Eksploratif (EDA), Data Cleaning, & Generator Data Sintetis        | **`/data`**          | [Baca README Data Science](./data/README.md)     |

---

## Fitur Utama

1. **AI Order Extraction**: Salin teks chat pembeli, dan model AI (FastAPI) akan otomatis mengekstrak _Nama Produk, Kuantitas, dan Harga_, lalu menghitung totalnya.
2. **Smart Fallback & Confidence Score**: Sistem mendeteksi otomatis jika AI kurang yakin (Confidence: LOW/MEDIUM) atau jika ada harga yang tidak disebutkan di chat, untuk dikonfirmasi ulang oleh kasir.
3. **TanyaAI (Asisten Bisnis UMKM)**: Chatbot terintegrasi berbasis Gemini 3.1 Flash yang dibatasi secara sistematis khusus untuk menjawab seputar strategi bisnis, keuangan, dan UMKM.
4. **Dashboard Analitik (Data Science)**: Visualisasi tren penjualan, performa produk, dan segmentasi harga menggunakan Streamlit.
5. **Decoupled Architecture**: Arsitektur dipisah antara Frontend, Main Backend (Node.js), dan AI API (Python) agar keandalan sistem tetap terjaga.

---

## Arsitektur Sistem & Struktur Repositori

Proyek ini dibangun menggunakan arsitektur _microservices_ dengan pembagian direktori sebagai berikut:

- 📂 **`frontend/`** — Antarmuka Web (UI) menggunakan **React, Vite, & Tailwind CSS v4**.
- 📂 **`backend/`** — Server REST API menggunakan **Node.js & Express**. Terhubung dengan **Supabase (PostgreSQL)** untuk database & autentikasi, serta bertindak sebagai _proxy_ aman untuk Gemini API.
- 📂 **`api-inference/`** — Server AI menggunakan **Python & FastAPI**. Bertugas memuat model AI (Keras) dan mengekstrak entitas dari teks. (Didesain untuk di-deploy ke Hugging Face Spaces).
- 📂 **`ai-model/`** — Ruang kerja Data Scientist. Berisi Jupyter Notebooks untuk proses _Training_, _Evaluation_, dan arsitektur model Neural Network.
- 📂 **`dashboard/`** — Dashboard analitik interaktif menggunakan **Streamlit** untuk visualisasi dataset dan simulasi model.
- 📂 **`data/`** & 📂 **`docs/`** — Berisi pipeline pengolahan data mentah/sintetis dan dokumen referensi seperti API Contract, Skema Database, dan Postman Collection.

---

## Teknologi yang Digunakan

- **Frontend**: React, Vite, Tailwind CSS, Axios, React-Markdown.
- **Backend**: Node.js, Express, Supabase JS Client, Google Generative AI SDK (Gemini).
- **AI & Data Science**: Python, TensorFlow/Keras, FastAPI, Streamlit, Pandas, Scikit-Learn.
- **Database & Auth**: Supabase (PostgreSQL).
- **Deployment**: Vercel (Frontend & Backend), Hugging Face Spaces (API Inference).

---

## Panduan Instalasi Lokal

Ikuti instruksi mendalam berikut untuk menjalankan seluruh aplikasi ChatKasir di komputer (lokal) Anda.

### Catatan Penting Sebelum Memulai:

_Secara default, instalasi lokal ini dikonfigurasi untuk menembak langsung ke **Hugging Face Spaces AI API** dan **Supabase Production** milik tim kami. Hal ini bertujuan agar Anda dapat menguji fungsionalitas penuh aplikasi tanpa perlu mengunduh resource Machine Learning yang sangat berat atau melakukan training model ulang di komputer Anda._

### 1. Persiapan Awal (Prerequisites)

Pastikan komputer Anda sudah terpasang perangkat lunak berikut:

- **Node.js** (Versi LTS `>= 18.x.x`) $\rightarrow$ [Unduh di sini](https://nodejs.org/)
- **Git** $\rightarrow$ [Unduh di sini](https://git-scm.com/)
- Kode Editor seperti **Visual Studio Code**

### 2. Kloning Repositori & Masuk ke Folder Utama

Buka Terminal / Command Prompt Anda, lalu jalankan perintah berikut:

```bash
git clone [https://github.com/reihanersaa/ChatKasir.git](https://github.com/reihanersaa/ChatKasir.git)
cd ChatKasir
```

### 3. Setup & Menjalankan Core Backend API

1. Buka terminal baru dan masuk ke folder backend:
   ```bash
   cd backend
   ```
2. Pasang seluruh dependencies Node.js yang dibutuhkan:
   ```bash
   npm install
   ```
3. Buat file `.env` di dalam root folder `backend/` (lihat `backend/.env.example`), lalu isi dengan variabel berikut (Hubungi tim pengembang ChatKasir untuk mendapatkan token kredensial asli):

   ```env
   PORT=3000
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_anon_key
   AI_API_URL=https://<alamat-cloud-ai>.hf.space
   AI_API_KEY=your_ai_api_key
   GEMINI_API_KEY=your_gemini_api_key
   FRONTEND_URL=http://localhost:5173

   ```

4. Jalankan server backend dalam mode pengembangan (development mode):

   ```bash
   npm run dev
   ```

   Jika berhasil, terminal akan memunculkan log dan server berjalan di url: http://localhost:3000

### 4. Setup & Menjalankan Frontend (User Interface)

Layanan frontend menyediakan antarmuka web interaktif tempat kasir melakukan aktivitas copy-paste teks pesanan.

1. Buka terminal baru lagi (jangan matikan terminal backend), lalu masuk ke folder `frontend`:
   ```bash
   cd frontend
   ```
2. Pasang seluruh library UI yang dibutuhkan:
   ```bash
   npm install
   ```
3. Buat file bernama `.env` di dalam root folder `frontend/` (`/frontend/.env`), lalu konfigurasikan arah API-nya ke server backend lokal yang baru saja Anda jalankan:
   ```
   VITE_API_URL=http://localhost:3000
   VITE_SUPABASE_URL=your_supabase_cloud_url
   VITE_SUPABASE_KEY=your_supabase_anon_public_key
   ```
4. Nyalakan server lokal frontend menggunakan Vite:
   ```bash
   npm run dev
   ```
5. Buka browser Anda dan akses aplikasi di: `http://localhost:5173`

---

## Dokumentasi API

Komunikasi data berupa format JSON antara Frontend, Backend, dan API Inference telah distandarisasi secara ketat demi menjaga konsistensi sistem.

- Detail struktur request dan response dapat dibaca di: [`docs/API_CONTRACT.md`]
- Tim pengembang juga menyediakan Postman Collection siap pakai di dalam folder docs/ untuk mempermudah pengujian endpoint.

---

## Pedoman Kontribusi (Contributing)

Kami menyambut kontribusi dari siapa saja untuk mengembangkan aplikasi ini menjadi lebih baik!

1. Lakukan _Fork_ pada repositori ini.
2. Buat _branch_ fitur Anda (`git checkout -b feature/FiturKeren`).
3. Lakukan _Commit_ perubahan Anda (`git commit -m 'Menambahkan fitur keren'`).
4. _Push_ ke _branch_ tersebut (`git push origin feature/FiturKeren`).
5. Buat sebuah _Pull Request_ (PR) baru ke branch utama kami.

---

## Tim Pengembang

Proyek ini dibangun secara kolaboratif sebagai bagian dari program **Coding Camp 2026 by DBS Foundation & Dicoding Indonesia**:

- **Alfan Ramadhan** - Front End Web Developer (FS-1)
- **Muhammad Reihan Ersa Putra** - Back End Web Developer (FS-2)
- **Muhammad Faradi Eka Damara** - Data Engineer / Data Science 1 (DS-1)
- **Salman Pandu Pandiya** - Data Analyst & Dashboard (DS-2)
- **Achmad Rif’an** - AI Model Architect (AI-1)
- **Denny Yoan Hendrawan** - AI API & Inference (AI-2)

---
