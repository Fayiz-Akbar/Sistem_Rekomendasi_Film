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
Dataset yang digunakan adalah **"The Movies Dataset"** yang diunduh dari Kaggle. Dataset ini berisi metadata untuk sekitar 45.000 film serta 26 juta rating dari pengguna.  
🔗 [https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

### Pemeriksaan Struktur Data

```python
# Gabungkan dulu movies_df dan ratings_df
movie_data = pd.merge(ratings_df, movies_df[['movieId', 'title', 'genres']], on='movieId', how='left')

movie_data.info()
movie_data.isnull().sum()
```

### Hasil Pemeriksaan

- Jumlah total baris setelah digabung: **26,025,358**
- Jumlah kolom: 5 (`userId`, `movieId`, `rating`, `title`, `genres`)
- Nilai kosong (`NaN`) pada kolom `title`: **14,587,721**
- Nilai kosong (`NaN`) pada kolom `genres`: **14,587,721**
- Data duplikat: Tidak ada

### Uraian Fitur Dataset

| Fitur     | Deskripsi                          | Tipe Data |
|----------|------------------------------------|-----------|
| `userId` | ID unik untuk setiap pengguna.     | int64     |
| `movieId`| ID unik untuk setiap film.         | int64     |
| `rating` | Rating yang diberikan (skala 0.5–5.0). | float64  |
| `title`  | Judul lengkap film.                | object    |
| `genres` | Genre film, dipisahkan oleh `|`.   | object    |

---

## Data Preparation

Tahapan persiapan data dilakukan untuk memastikan data bersih dan siap digunakan untuk pemodelan.

1. **Memuat & Menggabungkan Data**  
   Memuat `movies_metadata.csv` dan `ratings.csv`, lalu menggabungkannya menjadi satu DataFrame utama.

2. **Pembersihan Data Awal**  
   Membersihkan `movieId` yang tidak valid pada `movies_metadata.csv` untuk memastikan proses penggabungan berjalan lancar.

3. **Exploratory Data Analysis (EDA)**  
   Visualisasi:
   - **Distribusi Rating**: Grafik menunjukkan rating 4.0 adalah yang paling sering diberikan.  
     *(Masukkan gambar `rating_distribution.png` di sini)*
   - **Film Terpopuler**: Grafik menampilkan 10 film dengan jumlah rating terbanyak.  
     *(Masukkan gambar `top_10_movies.png` di sini)*

4. **Subset untuk Content-Based Filtering**  
   Membuat subset data `movie_features_subset` yang hanya berisi 15.000 film terpopuler.
   ```python
   movie_rating_counts = movie_data['movieId'].value_counts()
   top_movie_ids = movie_rating_counts.head(15000).index
   movie_features_subset = movie_features.loc[movie_features.index.isin(top_movie_ids)]
   ```

5. **Sampel untuk Collaborative Filtering**  
   Mengambil sampel 1 juta rating untuk melatih model SVD.
   ```python
   from surprise import Reader, Dataset
   ratings_sample = ratings_df.sample(n=1000000, random_state=42)
   reader = Reader(rating_scale=(1, 5))
   data_collaborative = Dataset.load_from_df(ratings_sample[['userId', 'movieId', 'rating']], reader)
   ```

---

## Modeling

### Model 1: Content-Based Filtering

- **TF-IDF Vectorizer** digunakan untuk mengubah teks genre menjadi vektor numerik.
- **Linear Kernel** digunakan untuk menghitung skor kemiripan antar vektor tersebut.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix_subset = tfidf.fit_transform(movie_features_subset['genres'])
cosine_sim_subset = linear_kernel(tfidf_matrix_subset, tfidf_matrix_subset)
```

### Model 2: Collaborative Filtering (SVD)

- Menggunakan algoritma **Singular Value Decomposition (SVD)** dari library `surprise`.
- Data dibagi menjadi 80% data latih dan 20% data uji.

```python
from surprise import SVD
from surprise.model_selection import train_test_split

trainset, testset = train_test_split(data_collaborative, test_size=0.2)
model_svd = SVD()
model_svd.fit(trainset)
```

---

## Evaluation

### Metrik Evaluasi

Metrik yang digunakan: **Precision@10**  
Metrik ini memastikan bahwa 10 film teratas yang direkomendasikan benar-benar relevan.

**Formula:**

$$\text{Precision@10} = \frac{|\text{Jumlah item relevan di 10 rekomendasi teratas}|}{10}$$

*Item dianggap "relevan" jika rating aslinya dari pengguna \>= 4.0.*

### Hasil Evaluasi

- **Precision@10**: **0.9230**

Ini berarti 92.3% dari film yang direkomendasikan memang disukai oleh pengguna (rating ≥ 4.0).

### Visualisasi Hasil

#### Content-Based untuk 'The Dark Knight':
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

#### Collaborative Filtering untuk User ID 1:
```
- The Million Dollar Hotel
```

> *Catatan: Model hanya menghasilkan satu rekomendasi karena sebagian besar film yang direkomendasikan tidak tersedia di metadata film.*

