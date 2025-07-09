import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD
from surprise.model_selection import train_test_split
from collections import defaultdict
from sklearn.metrics.pairwise import linear_kernel

movies_df = pd.read_csv('data/movies_metadata.csv', low_memory=False)

ratings_df = pd.read_csv('data/ratings.csv')

print("Dataset berhasil dimuat.")
print(f"Jumlah data film: {len(movies_df)}")
print(f"Jumlah data rating: {len(ratings_df)}")

# Hapus baris dengan ID yang tidak valid di movies_df
movies_df = movies_df[movies_df['id'].str.isnumeric()]

# Ubah tipe data kolom 'id' agar bisa digabungkan
movies_df['id'] = movies_df['id'].astype(int)

# Ganti nama kolom 'id' agar konsisten dengan 'movieId' di ratings_df
movies_df.rename(columns={'id': 'movieId'}, inplace=True)

# Gabungkan ratings_df dengan kolom terpilih dari movies_df
movie_data = pd.merge(ratings_df, movies_df[['movieId', 'title', 'genres']], on='movieId', how='left')

# Menampilkan 5 baris pertama dari data gabungan
print("\nContoh data setelah digabung:")
print(movie_data.head())

print("Movie_Data yang kosong berjumlah :", movie_data.isnull().sum())

movie_data.info()

sns.set(style='whitegrid')

# Hitung jumlah rating untuk setiap film
top_movies = movie_data['title'].value_counts().head(10)

print("10 Film dengan Rating Terbanyak:")
print(top_movies)

# Buat plot
plt.figure(figsize=(12, 6))
sns.barplot(x=top_movies.values, y=top_movies.index, palette='mako')
plt.title('Top 10 Film dengan Jumlah Rating Terbanyak', fontsize=16)
plt.xlabel('Jumlah Rating')
plt.ylabel('Judul Film')
plt.savefig('top_10_movies.png')
plt.show()

# Buat plot distribusi rating
plt.figure(figsize=(10, 5))
sns.countplot(x='rating', data=movie_data, palette='viridis')
plt.title('Distribusi Rating Film', fontsize=16)
plt.xlabel('Rating')
plt.ylabel('Jumlah Pengguna')
plt.savefig('rating_distribution.png')
plt.show()

# Membuat DataFrame baru yang hanya berisi fitur film
movie_features = movies_df[['movieId', 'title', 'genres']].copy()

# Hapus duplikat film berdasarkan movieId
movie_features.drop_duplicates(subset='movieId', inplace=True)

# Setel movieId sebagai index untuk pencarian yang lebih cepat nanti
movie_features.set_index('movieId', inplace=True)

print("Data untuk Content-Based Filtering (movie_features):")
print(movie_features.head())

# Hitung jumlah rating untuk setiap film
movie_rating_counts = movie_data['movieId'].value_counts()

# Pilih 15.000 film teratas dengan rating terbanyak
top_movie_ids = movie_rating_counts.head(15000).index

# Filter DataFrame movie_features agar hanya berisi film-film populer
movie_features_subset = movie_features.loc[movie_features.index.isin(top_movie_ids)]

print(f"Menggunakan subset data dengan {len(movie_features_subset)} film populer.")

# Mengambil sampel 1 juta rating untuk mempercepat proses
ratings_sample = ratings_df.sample(n=1000000, random_state=42)

# Membuat Reader untuk membaca skala rating dari 1 hingga 5
reader = Reader(rating_scale=(1, 5))

# Memuat data dari DataFrame ke format dataset surprise
# Pastikan urutan kolom adalah: user, item, rating
data_collaborative = Dataset.load_from_df(ratings_sample[['userId', 'movieId', 'rating']], reader)

print("Variabel 'data_collaborative' berhasil dibuat dan siap digunakan.")

# Membuat matriks TF-IDF dari subset
movie_features_subset['genres'] = movie_features_subset['genres'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix_subset = tfidf.fit_transform(movie_features_subset['genres'])

cosine_sim_subset = linear_kernel(tfidf_matrix_subset, tfidf_matrix_subset)

print("Matriks kemiripan untuk subset berhasil dibuat")

def get_content_based_recommendations(title, cosine_sim_matrix=cosine_sim_subset, movie_features=movie_features_subset):
    indices = pd.Series(movie_features.index, index=movie_features['title']).drop_duplicates()
    try:
        idx = indices[title]
        local_idx = movie_features.index.get_loc(idx)
    except KeyError:
        return f"Film dengan judul '{title}' tidak ditemukan atau tidak termasuk dalam subset populer."

    sim_scores = list(enumerate(cosine_sim_matrix[local_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    
    movie_indices = [i[0] for i in sim_scores]
    
    # Mengembalikan judul dan genre
    return movie_features[['title', 'genres']].iloc[movie_indices]

# Dapatkan rekomendasi untuk salah satu film populer
recommendations = get_content_based_recommendations('The Dark Knight')
print("Rekomendasi Content-Based untuk 'The Dark Knight':")
print(recommendations)

# Fungsi untuk menghitung Precision@k
def precision_recall_at_k(predictions, k=10, threshold=4.0):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
    
    return sum(prec for prec in precisions.values()) / len(precisions)

# Bagi data menjadi training dan testing
trainset, testset = train_test_split(data_collaborative, test_size=0.2)

# Inisialisasi dan latih model SVD
model_svd = SVD()
model_svd.fit(trainset)

# Lakukan prediksi pada test set
predictions = model_svd.test(testset)

# Hitung dan cetak Precision@10
precision_at_10 = precision_recall_at_k(predictions, k=10, threshold=4.0)
print(f"\nPrecision@10 untuk model SVD: {precision_at_10:.4f}")

def calculate_content_based_precision(ratings_df, movie_features_subset, cosine_sim_matrix):
    # Ambil sampel 500 user yang aktif untuk evaluasi
    active_users = ratings_df['userId'].value_counts().head(500).index
    
    precisions = []

    for user_id in active_users:
        # 1. Ambil film yang disukai user (ground truth)
        liked_movies = ratings_df[(ratings_df['userId'] == user_id) & (ratings_df['rating'] >= 4.0)]
        
        # Pastikan film yang disukai ada di dalam subset kita
        liked_movies_in_subset = liked_movies[liked_movies['movieId'].isin(movie_features_subset.index)]
        
        if liked_movies_in_subset.empty:
            continue

        # 2. Ambil satu film sebagai input untuk rekomendasi
        input_movie_id = liked_movies_in_subset.iloc[0]['movieId']
        
        # Cari judul film berdasarkan movieId
        try:
            input_movie_title = movie_features_subset.loc[input_movie_id]['title']
        except KeyError:
            continue

        # 3. Hasilkan rekomendasi
        recommendations = get_content_based_recommendations(input_movie_title, cosine_sim_matrix, movie_features_subset)
        if isinstance(recommendations, str): # Handle jika film tidak ditemukan
            continue
            
        recommended_movie_ids = movie_features_subset[movie_features_subset['title'].isin(recommendations['title'])].index
        
        # 4. Hitung item yang relevan dan direkomendasikan
        # Kita kecualikan film input itu sendiri dari perhitungan
        relevant_items = set(liked_movies_in_subset['movieId']) - {input_movie_id}
        relevant_and_recommended = set(recommended_movie_ids) & relevant_items
        
        # 5. Hitung precision untuk user ini
        # Pastikan panjang rekomendasi tidak nol untuk menghindari pembagian dengan nol
        if len(recommendations) > 0:
            precision = len(relevant_and_recommended) / len(recommendations)
            precisions.append(precision)

    # Rata-rata presisi dari semua user yang dievaluasi
    return sum(precisions) / len(precisions) if precisions else 0

# Jalankan evaluasi
cbf_precision = calculate_content_based_precision(ratings_df, movie_features_subset, cosine_sim_subset)

# Cetak hasilnya untuk dimasukkan ke laporan
print(f"Precision@10 untuk Content-Based Filtering: {cbf_precision:.4f}")

def get_collaborative_recommendations(userId, model=model_svd, ratings_df=ratings_df, movie_features=movie_features):
    # Mendapatkan daftar semua movieId unik
    all_movie_ids = ratings_df['movieId'].unique()
    
    # Mendapatkan daftar film yang sudah ditonton oleh user
    movies_watched_by_user = ratings_df[ratings_df['userId'] == userId]['movieId']
    
    # Dapatkan daftar film yang BELUM ditonton oleh user
    movies_to_predict = np.setdiff1d(all_movie_ids, movies_watched_by_user)
    
    # Lakukan prediksi untuk film yang belum ditonton
    testset_for_user = [[userId, movie_id, 4.] for movie_id in movies_to_predict]
    predictions = model.test(testset_for_user)
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Ambil top 10 movieId
    top_10_movie_ids = [pred.iid for pred in predictions[:10]]
    
    # Memberi filter ID yang hanya ada di dalam indeks movie_features
    existing_ids = [movie_id for movie_id in top_10_movie_ids if movie_id in movie_features.index]

    top_10_movies = movie_features.loc[existing_ids]['title']
    return top_10_movies

# Dapatkan rekomendasi untuk user dengan ID 1
collab_recommendations = get_collaborative_recommendations(userId=1)
print("\nRekomendasi Collaborative Filtering untuk User ID 1:")
print(collab_recommendations)