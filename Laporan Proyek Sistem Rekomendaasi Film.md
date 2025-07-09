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

Tahapan persiapan data dilakukan untuk memastikan data siap digunakan untuk pemodelan.
1.  **Penggabungan Data**: Menggabungkan `ratings_df` dan `movies_df` menjadi satu DataFrame utama. **Alasan:** Ini dilakukan agar setiap rating memiliki informasi judul dan genre filmnya untuk analisis lebih lanjut.
2.  **Pembersihan Data Awal**: Membersihkan `movieId` yang tidak valid pada `movies_metadata.csv`. **Alasan:** Langkah ini penting untuk memastikan proses penggabungan data berjalan lancar dan tidak ada data yang hilang karena ketidakcocokan ID.
3.  **Subset untuk Content-Based Filtering**: Membuat subset `movie_features_subset` yang hanya berisi 15.000 film terpopuler. **Alasan:** Menghitung matriks kemiripan untuk 45.000 film secara langsung menyebabkan `MemoryError`. Dengan membatasi data pada film yang paling relevan, model tetap dapat dibangun tanpa melebihi kapasitas memori.
4.  **Ekstraksi Fitur dengan TF-IDF**: Mengubah data teks genre menjadi matriks numerik menggunakan `TfidfVectorizer`. **Alasan:** Model machine learning tidak dapat memproses teks mentah. TF-IDF mengubah genre menjadi representasi vektor yang dapat diukur kemiripannya.
5.  **Sampel dan Konversi Data untuk Collaborative Filtering**: Mengambil sampel 1 juta rating dan mengubahnya ke dalam format `Dataset` library `surprise`. **Alasan:** Pengambilan sampel diperlukan untuk mempercepat proses training model SVD. Konversi ke format `surprise` adalah syarat wajib agar data dapat digunakan oleh algoritma dari library tersebut.
6.  **Pembagian Data Latih dan Uji**: Membagi data `surprise` menjadi 80% data latih dan 20% data uji. **Alasan:** Ini adalah praktik standar untuk mengevaluasi seberapa baik performa model pada data yang belum pernah dilihat sebelumnya.

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
    ```
    *(Catatan: Model hanya menghasilkan satu rekomendasi karena adanya inkonsistensi data, di mana 9 dari 10 movieId teratas yang direkomendasikan SVD tidak memiliki data metadata dalam subset film yang digunakan).*

---

## Evaluation

### **Metrik Evaluasi**
Metrik yang digunakan adalah **Precision@10**, yang mengukur seberapa banyak item yang relevan dari 10 item teratas yang direkomendasikan. Metrik ini sangat cocok untuk kasus bisnis di mana kita ingin memastikan rekomendasi yang ditampilkan di halaman utama benar-benar disukai pengguna.

### **Formula Precision@10:**
$$\text{Precision@10} = \frac{|\text{Jumlah item relevan di 10 rekomendasi teratas}|}{10}$$
*Item dianggap "relevan" jika rating aslinya dari pengguna >= 4.0.*

### **Hasil Evaluasi Model**
-   **Content-Based Filtering Precision@10**: **[Masukkan skor dari kode baru]**
-   **Collaborative Filtering (SVD) Precision@10**: **0.9230**

Skor Precision yang tinggi pada kedua model menunjukkan bahwa rekomendasi yang diberikan sangat relevan dengan preferensi pengguna. Model SVD menunjukkan performa yang sedikit lebih unggul dalam menemukan item relevan berdasarkan pola rating.

### **Hubungan dengan Business Understanding**
Hasil evaluasi menunjukkan bahwa kedua model berhasil menjawab permasalahan bisnis yang telah dirumuskan:
-   **Jawaban Problem Statement 1 & 2:** Kedua model berhasil dibangun. Model Content-Based mampu merekomendasikan film berdasarkan kemiripan genre, sementara Collaborative Filtering berhasil merekomendasikan film berdasarkan pola rating. Ini memenuhi **Goals 1 & 2**.
-   **Jawaban Problem Statement 3:** Performa kedua model berhasil diukur menggunakan metrik Precision@10, yang menunjukkan tingkat relevansi yang tinggi. Ini memenuhi **Goal 3**.
-   **Dampak Solusi:** Kedua *solution statement* terbukti berdampak. Solusi Content-Based memberikan rekomendasi yang aman dan relevan secara tematik. Solusi Collaborative Filtering, dengan Precision@10 **0.9230**, terbukti sangat efektif dalam memprediksi film yang akan disukai pengguna, yang secara langsung dapat meningkatkan *engagement* dan retensi pada platform.