import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def hitung_persentase_bobot(sum_true_input, sum_pattern):
    return (sum_true_input / sum_pattern) * 100 if sum_pattern else 0

def fuzzy_clustering(data, n_clusters, m=2, error=0.0001, maxiter=100):

    np.random.seed(42)  # Set seed agar hasil FCM stabil

    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        data.T, n_clusters, m, error=error, maxiter=maxiter, init=None
    )
    cluster_labels = np.argmax(u, axis=0)
    silhouette = silhouette_score(data, cluster_labels)
    return silhouette, cntr, cluster_labels, u


def proses_clustering(dataset_path, save_path='static/clustering.png',
                      silhouette_path='static/silhouette.png',
                      centroid_path='static/centroids.xlsx',
                      df_cluster_stats_path='static/df_cluster_stats.xlsx'):

    ext = os.path.splitext(dataset_path)[1]
    df = pd.read_csv(dataset_path) if ext == '.csv' else pd.read_excel(dataset_path)

    # Tampilkan dataframe yang sudah dipilih kolomnya
    print(df)

    # Preprocessing
    replace_dict = {'class': {'kognitif': 0, 'afektif': 1, 'psikomotorik': 2}}
    df.replace(replace_dict, inplace=True)
    # Pilih kolom x1 sampai x68 dan kolom class
    columns_to_select = ['x' + str(i) for i in range(1, 69)] + ['class']
    df = df[columns_to_select]
    df = df.map(lambda x: str(x).strip() if isinstance(x, str) else x)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.drop_duplicates()
    df = df.apply(lambda col: col.fillna(col.mean()), axis=0)

    pattern_values = {
        "Thinking": 80,
        "Striving": 90,
        "Relating": 90,
        "Influencing": 80
    }

    thinking_columns = [f"x{i}" for i in range(1, 17)]
    striving_columns = [f"x{i}" for i in range(17, 35)]
    relating_columns = [f"x{i}" for i in range(35, 53)]
    influencing_columns = [f"x{i}" for i in range(53, 69)]

    df['Thinking'] = df[thinking_columns].sum(axis=1)
    df['Striving'] = df[striving_columns].sum(axis=1)
    df['Relating'] = df[relating_columns].sum(axis=1)
    df['Influencing'] = df[influencing_columns].sum(axis=1)

    for var in pattern_values:
        df[var] = df[var].apply(lambda x: round(hitung_persentase_bobot(x, pattern_values[var]), 2))

    final_df = df[['Thinking', 'Striving', 'Relating', 'Influencing', 'class']]
    feature_columns = ['Thinking', 'Striving', 'Relating', 'Influencing']

    # === Split data ===
    X_data = final_df[feature_columns].values
    Y_data = final_df['class'].values
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.15, random_state=42)

    # === Normalisasi hanya pada data training ===
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)   

    # Evaluasi jumlah cluster (pakai data train saja)
    silhouette_scores = []
    for n_clusters in range(2, 9):
        labels = fuzzy_clustering(X_train_scaled, n_clusters)[2]
        score = silhouette_score(X_train_scaled, labels)
        silhouette_scores.append(score)

    optimal_clusters = 3
    silhouette, cntr, labels, u = fuzzy_clustering(X_train_scaled, optimal_clusters)

    # Simpan plot silhouette score
    plt.figure(figsize=(7, 5))
    plt.plot(range(2, 9), silhouette_scores, 'go-', markersize=8)
    plt.xlabel("Jumlah Cluster")
    plt.ylabel("Silhouette Score")
    plt.title("Evaluasi Silhouette Score untuk Jumlah Cluster")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(silhouette_path)
    plt.close()

    # Simpan plot clustering
    plt.figure(figsize=(6, 5))
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=labels, cmap='viridis', s=50)
    plt.scatter(cntr[:, 0], cntr[:, 1], c='red', s=100, marker='s')
    plt.title(f'Clustering dengan {optimal_clusters} Cluster\nSilhouette Score: {silhouette:.4f}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # ------------ Membership probability untuk setiap data ke masing-masing cluster pada data train -------------
    membership_probabilities = u.T  # (n_samples, n_clusters)

    # Ubah X_train ke DataFrame jika masih numpy array
    selected_features = ['Thinking', 'Striving', 'Relating', 'Influencing']
    X_train_df = pd.DataFrame(X_train, columns=selected_features)

    # Buat DataFrame untuk melihat probabilitas ke setiap cluster
    columns = ['Cluster_0', 'Cluster_1', 'Cluster_2']
    df_prob = pd.DataFrame(membership_probabilities, columns=columns)

    # Reset index agar gabungannya rapi
    X_train_df.reset_index(drop=True, inplace=True)
    df_prob.reset_index(drop=True, inplace=True)

    # Gabungkan fitur dan probabilitas cluster
    df_result = pd.concat([X_train_df, df_prob], axis=1)

    # Tampilkan hasil
    print(df_result.head(10))

    from scipy.spatial.distance import cdist

    # ----------------- Hitung jarak X_test_scaled ke masing-masing centroid dari X_train ------------------
    distances = cdist(X_test_scaled, cntr, metric='euclidean')

    # Hindari pembagian dengan nol
    distances = np.fmax(distances, np.finfo(np.float64).eps)

    # Hitung membership probability (harus sama nilai m dengan training)
    m = 2
    exponent = 2 / (m - 1)
    inv_distances = distances ** (-exponent)
    membership_test = inv_distances / np.sum(inv_distances, axis=1, keepdims=True)

    # ------------ Membership probability untuk setiap data ke masing-masing cluster pada data test -------------

    # Ubah X_test (belum dinormalisasi) ke DataFrame biar interpretasi fitur tetap asli
    selected_features = ['Thinking', 'Striving', 'Relating', 'Influencing']
    X_test_df = pd.DataFrame(X_test, columns=selected_features)

    # Ubah membership jadi DataFrame
    df_prob_test = pd.DataFrame(membership_test, columns=['Cluster_0', 'Cluster_1', 'Cluster_2'])

    # Gabungkan
    X_test_df.reset_index(drop=True, inplace=True)
    df_prob_test.reset_index(drop=True, inplace=True)
    df_result_test = pd.concat([X_test_df, df_prob_test], axis=1)

    # Tampilkan
    print(df_result_test.head(10))

    # Centroid dan probabilitas
    centroid_df = pd.DataFrame(cntr, columns=feature_columns)
    centroid_df.index = [f'Cluster {i+1}' for i in range(optimal_clusters)]

    # MEAN & STD
    selected_features = ['Thinking', 'Striving', 'Relating', 'Influencing']
    n_clusters = membership_probabilities.shape[1]
    data_asli = X_train_scaled  # Sudah ternormalisasi
    
    # --- Hitung weighted mean & std per cluster ---
    cluster_stats = {}

    for i in range(n_clusters):
        weights = membership_probabilities[:, i]  # (462,)
        stats = {}
    
        for j, feature in enumerate(selected_features):
            values = data_asli[:, j]  # (462,)

            # Pastikan semuanya float dan 1D
            values = np.array(values).reshape(-1).astype(float)
            weights = np.array(weights).reshape(-1).astype(float)
                
            # Weighted mean
            weighted_mean = np.average(values, weights=weights)

            # Weighted std deviasi
            weighted_std = np.sqrt(np.average((values - weighted_mean) ** 2, weights=weights))
                
            stats[f'{feature}_mean'] = weighted_mean
            stats[f'{feature}_std'] = weighted_std

        cluster_stats[f'Cluster_{i}'] = stats

    df_cluster_stats = pd.DataFrame(cluster_stats).T  # T: biar cluster jadi baris
    print(df_cluster_stats)

    # Gabung data train/test untuk sheet split
    split_df = pd.DataFrame(np.hstack((X_train, Y_train.reshape(-1, 1))), columns=feature_columns + ['class'])
    split_test_df = pd.DataFrame(np.hstack((X_test, Y_test.reshape(-1, 1))), columns=feature_columns + ['class'])
    split_combined = pd.concat([split_df.assign(Split='Train'), split_test_df.assign(Split='Test')])

    # Simpan ke Excel
    with pd.ExcelWriter('static/df.xlsx', engine='openpyxl') as writer:
        df_result.to_excel(writer, sheet_name='x_train', index=False)
        df_result_test.to_excel(writer, sheet_name='x_test', index=False)
        split_combined.to_excel(writer, sheet_name='split', index=False)

    # Simpan centroid
    centroid_df = centroid_df.round(6)
    centroid_df.to_excel(centroid_path, sheet_name='centroids', index=True)

    # Simpan Std dan Mean Probabilitas Cluster setiap variabel
    df_cluster_stats.to_excel(df_cluster_stats_path, sheet_name='cluster_stats', index=True)

    return silhouette, centroid_df, df_result, df_cluster_stats, labels, u
