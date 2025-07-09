# Laporan Proyek Machine Learning: Sistem Rekomendasi Film - Fayiz Akbar Daifullah

## Domain Proyek

Industri perfilman global saat ini dibanjiri oleh ribuan konten baru setiap tahun, menciptakan katalog yang sangat luas bagi platform *streaming* dan bioskop. Bagi pengguna, banyaknya pilihan ini sering kali menimbulkan ***choice paralysis***, di mana mereka menghabiskan lebih banyak waktu untuk mencari film daripada menikmatinya. Kondisi ini dapat menurunkan kepuasan dan keterlibatan (*engagement*) pengguna pada platform tersebut.

Untuk mengatasi masalah ini, sistem rekomendasi menjadi fitur krusial. Dengan menganalisis preferensi pengguna dan atribut film, sistem ini dapat menyajikan daftar tontonan yang dipersonalisasi, relevan, dan tepat waktu. Proyek ini bertujuan untuk membangun dua jenis sistem rekomendasi film yang dapat membantu pengguna menemukan film yang mereka sukai dengan lebih efisien, sehingga meningkatkan pengalaman menonton mereka secara keseluruhan.

> **Referensi:**
> 1. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. *Computer*, 42(8), 30-37.
> 2. Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS)*, 5(4), 1-19.

---

## Business Understanding

### Problem Statements
- Bagaimana cara membangun model yang dapat merekomendasikan film kepada pengguna berdasarkan kemiripan konten (genre) film yang pernah mereka sukai?
- Bagaimana cara memanfaatkan data rating historis untuk merekomendasikan film berdasarkan pola selera dari pengguna lain yang serupa?
- Bagaimana cara mengukur tingkat relevansi dari rekomendasi yang diberikan oleh model berbasis rating?

### Goals
- Mengembangkan model **Content-Based Filtering** untuk memberikan rekomendasi berdasarkan genre film.
- Mengembangkan model **Collaborative Filtering** untuk memberikan rekomendasi berdasarkan rating pengguna.
- Mengevaluasi performa model Collaborative Filtering menggunakan metrik **Precision@10**.

### Solution Statements
Proyek ini mengusulkan dua pendekatan solusi untuk membangun sistem rekomendasi:

1. **Menggunakan Content-Based Filtering dengan TF-IDF.** Metode ini akan menganalisis teks pada kolom `genres` untuk menemukan film-film yang memiliki kemiripan konten. Solusi ini cepat, mudah diinterpretasikan, dan tidak memerlukan data pengguna lain.
2. **Menggunakan Collaborative Filtering dengan SVD.** Metode ini akan menganalisis pola tersembunyi dalam data rating pengguna untuk menemukan preferensi. Algoritma **Singular Value Decomposition (SVD)** dipilih karena kemampuannya yang terbukti andal dalam kasus sistem rekomendasi berbasis rating. Performa solusi ini akan diukur secara kuantitatif dengan metrik **Precision@10** untuk memastikan relevansi rekomendasinya.

---

## Data Understanding

### Sumber Dataset
### **1. Dataset Metadata Film (`movies_metadata.csv`)**
-   **URL Sumber Data:** [https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
-   **Jumlah Data:** Terdiri dari **45.466 baris** dan **24 kolom**.
-   **Kondisi Data:** Terdapat beberapa nilai yang hilang di berbagai kolom, namun kolom yang akan digunakan (`id`, `title`, `genres`) relatif bersih. Tidak ditemukan data duplikat yang signifikan.
-   **Uraian Fitur yang Digunakan:**
    -   `id`: ID unik untuk setiap film.
    -   `title`: Judul lengkap film.
    -   `genres`: Genre film dalam format JSON.

### **2. Dataset Rating (`ratings.csv`)**
-   **URL Sumber Data:** [https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
-   **Jumlah Data:** Terdiri dari **26.024.289 baris** dan **4 kolom**.
-   **Kondisi Data:** Data rating tergolong lengkap, tidak memiliki nilai yang hilang pada kolom-kolom utamanya.
-   **Uraian Fitur:**
    -   `userId`: ID unik untuk setiap pengguna.
    -   `movieId`: ID unik film yang diberi rating.
    -   `rating`: Skor rating yang diberikan (skala 0.5-5.0).
    -   `timestamp`: Waktu pemberian rating.

---

## Data Preparation

Tahapan persiapan data dilakukan untuk memastikan data bersih, konsisten, dan siap digunakan untuk pemodelan. Proses ini dilakukan secara berurutan untuk menghindari kebocoran data dan memastikan setiap transformasi diterapkan dengan benar.

1.  **Pembersihan Data Awal (`movies_metadata.csv`)**
    Pada dataset metadata film, ditemukan beberapa baris di mana kolom `id` tidak berisi nilai numerik yang valid. Baris-baris ini dapat menyebabkan eror saat penggabungan data. Oleh karena itu, langkah pertama adalah menghapus baris-baris yang tidak valid ini. Selanjutnya, tipe data kolom `id` diubah menjadi integer (`int`) agar konsisten dengan `movieId` di dataset rating. Kolom ini juga diganti namanya menjadi `movieId` untuk memfasilitasi proses `merge`.
    ```python
    # Menghapus baris dengan ID yang tidak valid
    movies_df = movies_df[movies_df['id'].str.isnumeric()]
    # Mengubah tipe data kolom 'id'
    movies_df['id'] = movies_df['id'].astype(int)
    # Mengganti nama kolom
    movies_df.rename(columns={'id': 'movieId'}, inplace=True)
    ```

2.  **Penggabungan Data**
    Setelah pembersihan awal, DataFrame `ratings_df` dan `movies_df` digabungkan menjadi satu DataFrame utama. Alasan: Ini dilakukan agar setiap data rating memiliki informasi kontekstual seperti judul dan genre filmnya, yang krusial untuk analisis dan pemodelan.

3.  **Exploratory Data Analysis (EDA)**
    Proses ini dilakukan setelah penggabungan untuk mendapatkan insight dari data gabungan.
    -   **Distribusi Rating**: Grafik menunjukkan rating 4.0 adalah yang paling sering diberikan.
        *(Masukkan gambar `rating_distribution.png` di sini)*
    -   **Film Terpopuler**: Grafik menampilkan 10 film dengan jumlah rating terbanyak.
        *(Masukkan gambar `top_10_movies.png` di sini)*

4.  **Sampel dan Konversi Data untuk Collaborative Filtering**
    Sebelum data diproses untuk Content-Based, data untuk Collaborative Filtering disiapkan terlebih dahulu. Sampel acak sebanyak 1 juta rating diambil dari dataset utama. Alasan: Dataset rating asli sangat besar (26 juta baris), sehingga pengambilan sampel diperlukan untuk mempercepat proses training model SVD pada lingkungan dengan sumber daya terbatas. Data sampel ini kemudian dikonversi ke dalam format `Dataset` dari library `surprise`, karena ini adalah format wajib yang diterima oleh algoritma SVD.

5.  **Subset dan Ekstraksi Fitur untuk Content-Based Filtering**
    Langkah terakhir adalah mempersiapkan data untuk model Content-Based.
    -   **Subsetting**: Dibuat subset `movie_features_subset` yang hanya berisi 15.000 film terpopuler. Alasan: Menghitung matriks kemiripan untuk semua film secara langsung menyebabkan `MemoryError`. Dengan membatasi data pada film yang paling relevan (paling banyak diulas), model tetap dapat dibangun tanpa melebihi kapasitas memori.
    -   **Ekstraksi Fitur dengan TF-IDF**: Teks pada kolom `genres` diubah menjadi matriks numerik menggunakan `TfidfVectorizer`. Alasan: Model machine learning tidak dapat memproses teks mentah. TF-IDF mengubah genre menjadi representasi vektor yang dapat diukur kemiripannya.

---

## Modeling

### **Model 1: Content-Based Filtering**
-   **Definisi & Cara Kerja:** Model ini bekerja dengan prinsip "pengguna akan menyukai item yang mirip dengan item yang mereka sukai sebelumnya". Dalam proyek ini, kemiripan diukur berdasarkan **genre**. Prosesnya adalah mengubah setiap daftar genre film menjadi vektor numerik menggunakan TF-IDF. Kemudian, skor kemiripan antar setiap pasang film dihitung menggunakan **Linear Kernel**, yang secara matematis ekuivalen dengan Cosine Similarity untuk vektor ternormalisasi. Film dengan skor kemiripan tertinggi akan direkomendasikan.
-   **Hasil Top-N Rekomendasi (untuk 'The Dark Knight'):**
    ```
    - Carlito's Way
    - Killing Zoe
    - Romeo Is Bleeding
    - M
    - Mercury Rising
    - Payback
    - The Killing
    - Coogan's Bluff
    - The Way of the Gun
    - Best Seller
    ```

### **Model 2: Collaborative Filtering (SVD)**
-   **Definisi & Cara Kerja:** Model ini bekerja dengan prinsip "pengguna akan menyukai item yang juga disukai oleh pengguna lain dengan selera serupa". Algoritma **Singular Value Decomposition (SVD)** adalah teknik faktorisasi matriks. Ia menguraikan matriks interaksi pengguna-film yang besar menjadi dua matriks yang lebih kecil dan padat: matriks fitur laten pengguna dan matriks fitur laten film. Dengan mengalikan kembali kedua matriks ini, kita dapat memprediksi rating untuk film yang belum pernah dinilai oleh pengguna.
-   **Hasil Top-N Rekomendasi (untuk User ID 1):**
    ```
    - The Million Dollar Hotel
    - Once Were Warriors
    - Red River
    - Galaxy Quest
    - Cousin, Cousine

    ```
    *(Catatan: Model hanya menghasilkan satu rekomendasi karena adanya inkonsistensi data, di mana 9 dari 10 movieId teratas yang direkomendasikan SVD tidak memiliki data metadata dalam subset film yang digunakan).*

---

## Evaluation

### Metrik Evaluasi
Metrik yang digunakan untuk kedua model adalah **Precision@10**, yang mengukur seberapa banyak item yang relevan dari 10 item teratas yang direkomendasikan. Metrik ini sangat cocok untuk kasus bisnis di mana kita ingin memastikan rekomendasi yang ditampilkan kepada pengguna memiliki kualitas yang baik dan sesuai dengan selera mereka.

-   Untuk **Collaborative Filtering**, relevansi diukur berdasarkan apakah pengguna memberikan rating tinggi (>= 4.0) pada item yang direkomendasikan.
-   Untuk **Content-Based Filtering**, relevansi diukur dengan mengambil satu film yang disukai pengguna, membuat rekomendasi berdasarkan film tersebut, lalu melihat berapa banyak film yang direkomendasikan yang juga termasuk dalam daftar film yang disukai pengguna.

### Formula Precision@10:
$$\text{Precision@10} = \frac{|\text{Jumlah item relevan di 10 rekomendasi teratas}|}{10}$$
*Item dianggap "relevan" jika rating aslinya dari pengguna >= 4.0.*

### Hasil Evaluasi Model
-   **Content-Based Filtering Precision@10**: **0.1070**
-   **Collaborative Filtering (SVD) Precision@10**: **0.9253**

Hasil menunjukkan perbedaan performa yang signifikan. Model Collaborative Filtering (SVD) memiliki presisi yang sangat tinggi, membuktikan kemampuannya dalam menangkap preferensi personal pengguna. Sebaliknya, model Content-Based memiliki presisi yang lebih rendah. Hal ini wajar karena model ini hanya merekomendasikan berdasarkan kemiripan genre, yang bisa jadi tidak selalu sejalan dengan film lain yang disukai pengguna.

### Hubungan dengan Business Understanding
Hasil evaluasi menunjukkan bahwa kedua model berhasil menjawab permasalahan bisnis yang telah dirumuskan:
-   **Jawaban Problem Statement 1 & 2:** Kedua model berhasil dibangun. Model Content-Based mampu merekomendasikan film berdasarkan kemiripan genre, sementara Collaborative Filtering berhasil merekomendasikan film berdasarkan pola rating. Ini memenuhi **Goals 1 & 2**.
-   **Jawaban Problem Statement 3:** Performa kedua model berhasil diukur menggunakan metrik Precision@10, yang menunjukkan tingkat relevansi yang bervariasi. Ini memenuhi **Goal 3**.
-   **Dampak Solusi:** Kedua *solution statement* terbukti berdampak. Solusi Collaborative Filtering, dengan Precision@10 **0.9253**, terbukti sangat efektif dalam memprediksi film yang akan disukai pengguna, yang secara langsung dapat meningkatkan *engagement* dan retensi pada platform. Sementara itu, solusi Content-Based, meskipun dengan presisi kuantitatif yang lebih rendah (**0.1070**), tetap berhasil menyediakan rekomendasi yang relevan secara tematik dan dapat diandalkan saat data rating pengguna tidak tersedia.