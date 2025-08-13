import os
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate
from keras.saving import register_keras_serializable
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import keras

# Fungsi Keanggotaan Generalized Bell (GBell)
def generalized_bell_mf(x, a, b, c):
    return 1 / (1 + tf.pow(tf.abs((x - c) / a), 2 * b))

# Parameter keanggotaan untuk setiap variabel
# Ambil berdasarkan df_cluster_stats dengan susunan std, b, mean

params = {
    "Thinking": [(0.3, 2, 0.24), (0.3, 2, 0.51), (0.3, 2, 0.67)],
    "Striving": [(0.3, 2, 0.27), (0.3, 2, 0.51), (0.3, 2, 0.75)],
    "Relating": [(0.3, 2, 0.24), (0.3, 2, 0.435), (0.3, 2, 0.63)],
    "Influencing": [(0.3, 2, 0.34), (0.3, 2, 0.57), (0.3, 2, 0.71)],
}

# Fungsi untuk menerapkan fuzzifikasi pada setiap variabel
def apply_fuzzification(X, params):
    fuzzy_values = []
    for var, param_list in params.items():
        index = list(params.keys()).index(var)
        var_values = X[:, index]
        fuzzified_values = tf.stack(
            [generalized_bell_mf(var_values, a, b, c) for (a, b, c) in param_list], axis=1
        )
        fuzzy_values.append(fuzzified_values)
    return tf.concat(fuzzy_values, axis=1)


# Load data dari df.xlsx, sheet 'split'
file_path = os.path.join("static", "df.xlsx")
df = pd.read_excel(file_path, sheet_name="split")

# Ambil hanya kolom fitur yang dibutuhkan
features = ["Thinking", "Striving", "Relating", "Influencing"]
df_features = df[features]

# Nilai pattern untuk masing-masing variabel

# Bagi data berdasarkan kolom 'Split' menjadi Train dan Test
train_data = df[df['Split'] == 'Train']
test_data = df[df['Split'] == 'Test']

# Pisahkan fitur dan label untuk data Train dan Test
X_train = train_data[features].values.astype(np.float32)
Y_train = train_data['class'].values.astype(np.int32)

X_test = test_data[features].values.astype(np.float32)
Y_test = test_data['class'].values.astype(np.int32)

#Membuat variabel menampung nilai data asli dari X_train untuk digunakan saat prediksi data baru
X_train_processed = X_train
X_test_processed = X_test

# Normalisasi data
scaler = MinMaxScaler(feature_range=(0, 1))

# Normalisasi X_train dan ubah kembali ke DataFrame untuk menjaga kolom
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features)

# Normalisasi X_test dengan menggunakan fit_transform pada X_train (untuk menghindari data leakage)
X_test = pd.DataFrame(scaler.transform(X_test), columns=features)

#Membuat variabel menampung nilai data asli dari X_train untuk digunakan saat prediksi data baru
X_train = X_train.values
X_test = X_test.values

# Ubah numpy array ke DataFrame
X_train_processed_df = pd.DataFrame(X_train_processed)

# Simpan ke file Excel
X_train_processed_df.to_excel("static/X_train_processed.xlsx", index=False)
print("File Excel berhasil disimpan!")

# Simpan X_train dan X_test hasil normalisasi ke file Excel
X_train_norm_df = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
X_test_norm_df = pd.DataFrame(scaler.transform(X_test), columns=features)
X_train_norm_df.to_excel("static/X_train_norm.xlsx", index=False)
X_test_norm_df.to_excel("static/X_test_norm.xlsx", index=False)

# Jika diperlukan, bisa melakukan verifikasi atau pencetakan untuk memastikan
print(f"Data Train: {X_train.shape}, Data Test: {X_test.shape}")

# Konversi ke tensor
X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)

# Lakukan fuzzifikasi
X_train_fuzzy = apply_fuzzification(X_train_tensor, params)
X_test_fuzzy = apply_fuzzification(X_test_tensor, params)

# Buat nama kolom
category_mapping = {0: "Low", 1: "Medium", 2: "High"}
columns = [f"{var}_{category_mapping[i]}" for var in params.keys() for i in range(3)]

# Konversi hasil ke DataFrame
X_train_fuzzy_df = pd.DataFrame(X_train_fuzzy.numpy(), columns=columns)
X_test_fuzzy_df = pd.DataFrame(X_test_fuzzy.numpy(), columns=columns)

# Hitung firing strengths
firing_strengths = np.array(X_train_fuzzy_df)
firing_strengths_df = pd.DataFrame(firing_strengths, columns=columns)
print(firing_strengths[:5])  # Tampilkan 5 baris pertama

# Simpan DataFrame hasil fuzzifikasi dan firing strength ke Excel
output_path = os.path.join("static", "1. classification.xlsx")
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    X_train_fuzzy_df.to_excel(writer, sheet_name="X_train_fuzzy_df", index=False)
    X_test_fuzzy_df.to_excel(writer, sheet_name="X_test_fuzzy_df", index=False)
    firing_strengths_df.to_excel(writer, sheet_name="firing_strengths_1", index=False)

# Buat visualisasi fungsi keanggotaan
x = np.linspace(-0.5, 1.5, 200)
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
category_labels = ["Low", "Medium", "High"]

for ax, (var, clusters) in zip(axes.flat, params.items()):
    for i, (a, b, c) in enumerate(clusters):
        label = f"{var}_{category_labels[i]}"
        ax.plot(x, generalized_bell_mf(x, a, b, c), label=label)
    ax.set_title(var)
    ax.set_xlabel("Nilai (Normalized)")
    ax.set_ylabel("Membership Degree")
    ax.legend()
    ax.grid()

plt.tight_layout()
plot_path = os.path.join("static", "2. membership_functions.png")
plt.savefig(plot_path)

# Layer Fuzzy sebagai input ANFIS fuzzifikasi
def fuzzy_layer(inputs):
    memberships = []
    for i, var in enumerate(["Thinking", "Striving", "Relating", "Influencing"]):
        for (a, b, c) in params[var]:
            memberships.append(generalized_bell_mf(inputs[:, i], a, b, c))  # Fungsi Keanggotaan
    return tf.stack(memberships, axis=1)  # Gabungkan semua dalam satu tensor

# Input Layer (sesuai jumlah fitur asli, bukan yang sudah difuzzifikasi)
input_layer = keras.Input(shape=(X_train.shape[1],)) #karena X_train.shape hanya memiliki 2 dimensi


import tensorflow as tf
from keras.saving import register_keras_serializable

# ====== FUZZY MEMBERSHIP LAYER ======
@register_keras_serializable()
# def fuzzy_layer_tf(x):
#     a = tf.constant([1.0] * 12, dtype=tf.float32) #12 karena total fungsi keanggotaan ada 3 yaitu low, medium, high dan 4 variabel, so 4 x 3 =12
#     b = tf.constant([2.0] * 12, dtype=tf.float32)

#     # Atur c sesuai titik tengah fungsi Low, Medium, High
#     c = tf.constant([0.0, 0.5, 1.0] * 4, dtype=tf.float32)  # 3 membership × 4 fitur

#     x_exp = tf.expand_dims(x, axis=2)         # (batch, 4, 1)
#     x_tile = tf.tile(x_exp, [1, 1, 3])         # (batch, 4, 3)
#     x_flat = tf.reshape(x_tile, [-1, 12])      # (batch, 12)

#     result = 1 / (1 + tf.pow(tf.abs((x_flat - c) / a), 2 * b))
#     return result  # (batch, 11)

def fuzzy_layer_tf(x):
    a = tf.constant([0.3] * 12, dtype=tf.float32)
    b = tf.constant([2.0] * 12, dtype=tf.float32)  # tetap 2 untuk semua
    
    # Means (c), sesuai urutan sama dengan a, urutkan nilai dari low, medium, high
    c = tf.constant([
        0.24, 0.51, 0.67,
        0.27, 0.51, 0.75,
        0.24, 0.435, 0.63,
        0.34, 0.57, 0.71
    ], dtype=tf.float32)
    
    x_exp = tf.expand_dims(x, axis=2)         # (batch, 4, 1)
    x_tile = tf.tile(x_exp, [1, 1, 3])         # (batch, 4, 3)
    x_flat = tf.reshape(x_tile, [-1, 12])      # (batch, 12)
    
    result = 1 / (1 + tf.pow(tf.abs((x_flat - c) / a), 2 * b))
    return result

# ====== RULE LAYER ======
@register_keras_serializable()
def rule_layer(memberships, excluded_rules=None):

    tensors = tf.reshape(memberships, [-1, 12])

    # Rule definitions
    #Kognitif Rule
    rule1 = tf.reduce_min([tensors[:, 1], tensors[:, 3], tensors[:, 6], tensors[:, 9]], axis=0)   # medium low low low
    rule2 = tf.reduce_min([tensors[:, 2], tensors[:, 3], tensors[:, 6], tensors[:, 9]], axis=0)   # high low low low
    rule3 = tf.reduce_min([tensors[:, 2], tensors[:, 3], tensors[:, 6], tensors[:, 10]], axis=0)  # high low low medium
    rule4 = tf.reduce_min([tensors[:, 2], tensors[:, 3], tensors[:, 7], tensors[:, 9]], axis=0)   # high low medium low
    rule5 = tf.reduce_min([tensors[:, 2], tensors[:, 3], tensors[:, 7], tensors[:, 10]], axis=0)  # high low medium medium
    rule6 = tf.reduce_min([tensors[:, 2], tensors[:, 4], tensors[:, 6], tensors[:, 9]], axis=0)   # high medium low low
    rule7 = tf.reduce_min([tensors[:, 2], tensors[:, 4], tensors[:, 6], tensors[:, 10]], axis=0)  # high medium low medium
    rule8 = tf.reduce_min([tensors[:, 2], tensors[:, 4], tensors[:, 7], tensors[:, 9]], axis=0)   # high medium medium low
    rule9 = tf.reduce_min([tensors[:, 2], tensors[:, 4], tensors[:, 7], tensors[:, 10]], axis=0)  # high medium medium medium

    # Afektif Rule
    rule10 = tf.reduce_min([tensors[:, 0], tensors[:, 3], tensors[:, 6], tensors[:, 10]], axis=0) # low low low medium
    rule11 = tf.reduce_min([tensors[:, 0], tensors[:, 3], tensors[:, 6], tensors[:, 11]], axis=0) # low low low high
    rule12 = tf.reduce_min([tensors[:, 0], tensors[:, 3], tensors[:, 7], tensors[:, 10]], axis=0) # low low medium medium
    rule13 = tf.reduce_min([tensors[:, 0], tensors[:, 3], tensors[:, 7], tensors[:, 11]], axis=0) # low low medium high
    rule14 = tf.reduce_min([tensors[:, 0], tensors[:, 3], tensors[:, 8], tensors[:, 10]], axis=0) # low low high medium
    rule15 = tf.reduce_min([tensors[:, 0], tensors[:, 3], tensors[:, 8], tensors[:, 11]], axis=0) # low low high high
    rule16 = tf.reduce_min([tensors[:, 0], tensors[:, 4], tensors[:, 6], tensors[:, 11]], axis=0) # low medium low high
    rule17 = tf.reduce_min([tensors[:, 0], tensors[:, 4], tensors[:, 7], tensors[:, 11]], axis=0) # low medium medium high
    rule18 = tf.reduce_min([tensors[:, 0], tensors[:, 4], tensors[:, 8], tensors[:, 11]], axis=0) # low medium high high
    rule19 = tf.reduce_min([tensors[:, 1], tensors[:, 3], tensors[:, 6], tensors[:, 11]], axis=0) # medium low low high
    rule20 = tf.reduce_min([tensors[:, 1], tensors[:, 3], tensors[:, 7], tensors[:, 11]], axis=0) # medium low medium high
    rule21 = tf.reduce_min([tensors[:, 1], tensors[:, 3], tensors[:, 8], tensors[:, 10]], axis=0) # medium low high medium
    rule22 = tf.reduce_min([tensors[:, 1], tensors[:, 3], tensors[:, 8], tensors[:, 11]], axis=0) # medium low high high
    rule23 = tf.reduce_min([tensors[:, 1], tensors[:, 4], tensors[:, 6], tensors[:, 11]], axis=0) # medium medium low high
    rule24 = tf.reduce_min([tensors[:, 1], tensors[:, 4], tensors[:, 7], tensors[:, 11]], axis=0) #medium medium medium high
    rule25 = tf.reduce_min([tensors[:, 1], tensors[:, 4], tensors[:, 8], tensors[:, 11]], axis=0) # medium medium high high


    # Psikomotorik Rule
    rule26 = tf.reduce_min([tensors[:, 0], tensors[:, 4], tensors[:, 6], tensors[:, 9]], axis=0)  # low medium low low
    rule27 = tf.reduce_min([tensors[:, 0], tensors[:, 4], tensors[:, 7], tensors[:, 9]], axis=0)  # low medium medium low
    rule28 = tf.reduce_min([tensors[:, 0], tensors[:, 4], tensors[:, 8], tensors[:, 9]], axis=0)  # low medium high low
    rule29 = tf.reduce_min([tensors[:, 0], tensors[:, 5], tensors[:, 6], tensors[:, 9]], axis=0)  # low high low low
    rule30 = tf.reduce_min([tensors[:, 0], tensors[:, 5], tensors[:, 6], tensors[:, 10]], axis=0) # low high low medium
    rule31 = tf.reduce_min([tensors[:, 0], tensors[:, 5], tensors[:, 7], tensors[:, 9]], axis=0)  # low high medium low
    rule32 = tf.reduce_min([tensors[:, 0], tensors[:, 5], tensors[:, 7], tensors[:, 10]], axis=0) # low high medium medium
    rule33 = tf.reduce_min([tensors[:, 0], tensors[:, 5], tensors[:, 8], tensors[:, 9]], axis=0)  # low high high low
    rule34 = tf.reduce_min([tensors[:, 0], tensors[:, 5], tensors[:, 8], tensors[:, 10]], axis=0) # low high high medium
    rule35 = tf.reduce_min([tensors[:, 1], tensors[:, 4], tensors[:, 6], tensors[:, 9]], axis=0) # medium medium low low
    rule36 = tf.reduce_min([tensors[:, 1], tensors[:, 4], tensors[:, 7], tensors[:, 9]], axis=0) # medium medium medium low
    rule37 = tf.reduce_min([tensors[:, 1], tensors[:, 4], tensors[:, 8], tensors[:, 9]], axis=0) # medium medium high low
    rule38 = tf.reduce_min([tensors[:, 1], tensors[:, 5], tensors[:, 6], tensors[:, 9]], axis=0)  # medium high low low
    rule39 = tf.reduce_min([tensors[:, 1], tensors[:, 5], tensors[:, 6], tensors[:, 10]], axis=0) # medium high low medium
    rule40 = tf.reduce_min([tensors[:, 1], tensors[:, 5], tensors[:, 7], tensors[:, 9]], axis=0)  # medium high medium low
    rule41 = tf.reduce_min([tensors[:, 1], tensors[:, 5], tensors[:, 7], tensors[:, 10]], axis=0) # medium high medium medium
    rule42 = tf.reduce_min([tensors[:, 1], tensors[:, 5], tensors[:, 8], tensors[:, 9]], axis=0)  # medium high high low
    rule43 = tf.reduce_min([tensors[:, 1], tensors[:, 5], tensors[:, 8], tensors[:, 10]], axis=0) # medium high high medium
    rule44 = tf.reduce_min([tensors[:, 2], tensors[:, 5], tensors[:, 6], tensors[:, 9]], axis=0) #high high low low
    rule45 = tf.reduce_min([tensors[:, 2], tensors[:, 5], tensors[:, 6], tensors[:, 10]], axis=0)  # high high low medium
    rule46 = tf.reduce_min([tensors[:, 2], tensors[:, 5], tensors[:, 7], tensors[:, 9]], axis=0)  # high high medium low
    rule47 = tf.reduce_min([tensors[:, 2], tensors[:, 5], tensors[:, 7], tensors[:, 10]], axis=0)  # high high medium medium
    rule48 = tf.reduce_min([tensors[:, 2], tensors[:, 5], tensors[:, 8], tensors[:, 9]], axis=0)  # high high high low
    rule49 = tf.reduce_min([tensors[:, 2], tensors[:, 5], tensors[:, 8], tensors[:, 7]], axis=0)  # high high high medium

    # Gabungkan semua rule
    rule_strengths = tf.stack([
        rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11,
        rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20, rule21,
        rule22, rule23, rule24, rule25, rule26, rule27, rule28, rule29, rule30, rule31,
        rule32, rule33, rule34, rule35, rule36, rule37, rule38, rule39, rule40, rule41,
        rule42, rule43, rule44, rule45, rule46, rule47, rule48, rule49
    ], axis=1)

    # Mengecualikan aturan yang ada di `excluded_rules`
    if excluded_rules:
        # Mengubah excluded_rules menjadi 0-based (kura ngi 1 dari setiap indeks)
        excluded_rules_zero_based = [rule - 1 for rule in excluded_rules]

        # Menghapus aturan yang dikecualikan
        rule_strengths = tf.gather(rule_strengths, [i for i in range(rule_strengths.shape[1]) if i not in excluded_rules_zero_based], axis=1)


    return rule_strengths

# ====== RULE TO CLASS MAPPING (INDEX START FROM 1) ======
rule_groups = {
    'kognitif': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'afektif': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    'psikomotorik': [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
}

# Total rules
total_rules = 49
labels = [''] * total_rules
for class_name, rule_ids in rule_groups.items():
    for rule_idx in rule_ids:
        labels[rule_idx - 1] = class_name  # -1 karena Python pakai indeks mulai dari 0

# One-hot mapping
label_map = {
    'kognitif': [1.0, 0.0, 0.0],
    'afektif': [0.0, 1.0, 0.0],
    'psikomotorik': [0.0, 0.0, 1.0]
}
rule_classes = tf.constant([label_map[label] for label in labels], dtype=tf.float32)


# ====== BUILD ANFIS MODEL ======
input_tensor = tf.keras.Input(shape=(4,))  # Input 4 fitur

# Layer 1️⃣: Fuzzifikasi (ubah input menjadi nilai fuzzy)
fuzzy_output = tf.keras.layers.Lambda(fuzzy_layer_tf)(input_tensor)

# Layer 2️⃣: Rule Layer
rule_output = tf.keras.layers.Lambda(rule_layer)(fuzzy_output)

# Layer 3️⃣: Normalisasi dengan Softmax
T = 0.3
normalized_rules = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x / T))(rule_output) #Normalisasi aturan (softmax dengan temperature scaling)

# Layer 4️⃣ & 5️⃣: Consequent & Defuzzifikasi Layer
output_layer = tf.keras.layers.Lambda(lambda x: tf.matmul(x, rule_classes))(normalized_rules) #Manual Mapping

# Model definisi
anfis_model = tf.keras.Model(inputs=input_tensor, outputs=output_layer)
anfis_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
anfis_model.summary()

intermediate_model = tf.keras.Model(inputs=anfis_model.input, outputs=[fuzzy_output, rule_output, normalized_rules, output_layer])
intermediate_outputs = intermediate_model.predict(X_test[:1])  # 1 sample aja misal

fuzzy_out = intermediate_outputs[0]
rule_out = intermediate_outputs[1]
norm_out = intermediate_outputs[2]
final_out = intermediate_outputs[3]

# Buat model yang output-nya adalah normalized_rules (softmax hasil LSE)
rule_strength_model = tf.keras.Model(inputs=anfis_model.input, outputs=normalized_rules)

# Prediksi bobot LSE untuk sebagian data (misalnya 5 data pertama)
bobot_LSE = rule_strength_model.predict(X_train[:1])

# Tampilkan hasilnya
for i, row in enumerate(bobot_LSE):
    print(f"Sample {i+1} - Normalized Rule Strengths (setelah LSE):")
    print(np.round(row, 4))

# ====== Simpan Output ke dalam Excel ======
# Mengubah hasil output menjadi DataFrame
bobot_LSE_df = pd.DataFrame(bobot_LSE, columns=[f"Rule {i+1}" for i in range(bobot_LSE.shape[1])])

# Simpan ke dalam file Excel di folder static
bobot_LSE_df.to_excel("static/3. bobot_LSE_df.xlsx", index=False)


# One Hot Encoding
Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=3)

# Latih model tanpa validasi
anfis_model.fit(X_train, Y_train, epochs=1, batch_size=2, verbose=1)

# # Predict on test set
y_pred = anfis_model.predict(X_test)

# Ambil kelas dengan nilai tertinggi (one-hot encoding ke kategori)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = Y_test

# ====== HITUNG AKURASI ======
accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f"Akurasi Model: {accuracy:.4f}")

# ====== SIMPAN AKURASI KE DALAM FILE EXCEL ======
# Buat DataFrame untuk akurasi
accuracy_df = pd.DataFrame({
    'Metric': ['Accuracy'],
    'Value': [accuracy]
})

# Simpan ke dalam file Excel di folder static
accuracy_df.to_excel("static/4. accuracy_df.xlsx", index=False)

# ====== CONFUSION MATRIX ======
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Plot Heatmap dari confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1", "Class 2"], yticklabels=["Class 0", "Class 1", "Class 2"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")

# Menyimpan heatmap confusion matrix sebagai gambar di folder static
plt.savefig("static/5. confusion_matrix_heatmap.png")
plt.close()

# Tampilkan confusion matrix untuk referensi
print("Confusion Matrix:")
print(cm)

# ====== CLASSIFICATION REPORT ======
report = classification_report(y_true_classes, y_pred_classes, target_names=["Class 0", "Class 1", "Class 2"], zero_division=1)

# Tampilkan classification report di console
print("\nClassification Report:\n", report)

# ====== SIMPAN CLASSIFICATION REPORT KE DALAM FILE EXCEL ======
# Convert classification report ke DataFrame
report_dict = classification_report(y_true_classes, y_pred_classes, target_names=["Class 0", "Class 1", "Class 2"], zero_division=1, output_dict=True)

# Convert report_dict ke DataFrame
report_df = pd.DataFrame(report_dict).transpose()

# Simpan classification report ke dalam Excel di folder static
report_df.to_excel("static/6. report_df.xlsx")

# Fungsi untuk mendapatkan kategori untuk setiap variabel berdasarkan nilai tertinggi
def categorize_data_by_membership(df, variables, category_mapping):
    # Menyimpan hasil kategorisasi per baris
    categorized_data = []

    for _, row in df.iterrows():
        result = {}

        # Untuk setiap variabel, cari kolom dengan nilai tertinggi (Low, Medium, High)
        for var in variables:
            low_col = f"{var}_{category_mapping[0]}"
            medium_col = f"{var}_{category_mapping[1]}"
            high_col = f"{var}_{category_mapping[2]}"

            # Menentukan kategori yang memiliki nilai tertinggi
            max_category_idx = row[[low_col, medium_col, high_col]].idxmax()
            max_category = max_category_idx.split('_')[1]  # Ambil kategori (Low, Medium, High)
            result[f"{var}_Category"] = max_category
            result[f"{var}_Value"] = row[[low_col, medium_col, high_col]].max()  # Nilai tertinggi

        categorized_data.append(result)

    return categorized_data

# Terapkan fungsi ke DataFrame hasil fuzzifikasi
categorized_train_data = categorize_data_by_membership(X_train_fuzzy_df, params.keys(), category_mapping)

# Konversi hasil ke dalam DataFrame untuk tampilan yang lebih rapi
categorized_train_df = pd.DataFrame(categorized_train_data)

# Tampilkan 10 baris pertama untuk melihat hasilnya
print(categorized_train_df.head(10))

# ====== SIMPAN HASIL KATEGORIZASI KE DALAM FILE EXCEL ======
categorized_train_df.to_excel("static/7. categorized_train_data.xlsx", index=False)


# Buat DataFrame untuk melihat hasil klasifikasi
classification_results = pd.DataFrame(X_test_processed, columns=['Thinking', 'Striving', 'Relating', 'Influencing'])

# Tambahkan kolom kategori aktual dan prediksi
classification_results['Actual Class'] = y_true_classes
classification_results['Predicted Class'] = y_pred_classes

# Tampilkan 10 baris pertama hasil klasifikasi
from tabulate import tabulate
print(tabulate(classification_results.head(10), headers='keys', tablefmt='fancy_grid'))

# ====== SIMPAN HASIL KLASIFIKASI KE DALAM FILE EXCEL ======
classification_results.to_excel("static/8. classification_results.xlsx", index=False)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate

# Fungsi untuk mendapatkan kategori berdasarkan nilai tertinggi dari GBell MF
def get_membership_category(value, param_list):
    memberships = [generalized_bell_mf(value, *params) for params in param_list]
    categories = ['Low', 'Medium', 'High']
    return categories[np.argmax(memberships)]  # Pilih kategori dengan nilai tertinggi

# ***1️⃣ Normalisasi Data X_test***
scaler = MinMaxScaler(feature_range=(0, 1))  # Normalisasi 0-1
X_test_normalized = scaler.fit_transform(X_test)  # Transformasikan data

# ***2️⃣ Konversi X_test Normalisasi ke DataFrame***
X_test_df = pd.DataFrame(X_test_normalized, columns=['Thinking', 'Striving', 'Relating', 'Influencing'])

# ***3️⃣ Loop untuk mengkategorikan setiap variabel berdasarkan GBell MF***
for var in ['Thinking', 'Striving', 'Relating', 'Influencing']:
    X_test_df[f'{var}_category'] = X_test_df[var].apply(lambda x: get_membership_category(x, params[var]))

# ***4️⃣ Tambahkan kolom kelas asli dan hasil prediksi***
X_test_df['Actual Class'] = y_true_classes
X_test_df['Predicted Class'] = y_pred_classes

# ***5️⃣ Tampilkan hasil dalam format tabel***
print(tabulate(X_test_df, headers='keys', tablefmt='fancy_grid'))

# ====== SIMPAN HASIL KE DALAM FILE EXCEL ======
X_test_df.to_excel("static/9. X_test_df.xlsx", index=False)

#LABELLING

# Buat DataFrame dari X_test
classified_train_df = pd.DataFrame(X_test, columns=['Thinking', 'Striving', 'Relating', 'Influencing'])

# Pastikan y_true_classes dan y_pred_classes memiliki panjang yang sama dengan X_test
if len(y_true_classes) != len(X_test) or len(y_pred_classes) != len(X_test):
    raise ValueError("Ukuran y_true_classes atau y_pred_classes tidak sesuai dengan X_test")

# Tambahkan label asli dan hasil prediksi
classified_train_df['True_Label'] = y_true_classes
classified_train_df['Predicted_Label'] = y_pred_classes

# Mapping kategori dari angka ke label
label_mapping = {0: 'Kognitif', 1: 'Afektif', 2: 'Psikomotorik'}
classified_train_df['True_Label'] = classified_train_df['True_Label'].map(label_mapping)
classified_train_df['Predicted_Label'] = classified_train_df['Predicted_Label'].map(label_mapping)

# Tampilkan hasil dalam format tabel yang rapi
print(tabulate(classified_train_df.head(10), headers='keys', tablefmt='fancy_grid'))

# ====== SIMPAN HASIL KE EXCEL ======
# Simpan classified_train_df ke file Excel dalam folder static
classified_train_df.to_excel("static/10. classified_train_df.xlsx", index=False)

rule_model = tf.keras.Model(inputs=anfis_model.input, outputs=rule_output)
rule_output_val = rule_model.predict(X_test)  # hasil shape: (n_samples, n_rules)

# ====== SIMPAN HASIL KE EXCEL ======
# Mengonversi rule_output_val ke DataFrame agar bisa disimpan ke Excel
rule_output_df = pd.DataFrame(rule_output_val, columns=[f"Rule {i+1}" for i in range(rule_output_val.shape[1])])

# Simpan ke Excel
rule_output_df.to_excel("static/11. rule_output_df.xlsx", index=False)
print(rule_output_val)

# ====== EVALUASI RULE MENGGUNAKAN COSINE SIMILARITY ======

# Representasi numerik dari setiap rule (49 rule × 4 variabel input)
# Format tiap rule: [variabel1, variabel2, variabel3, variabel4]
# 0: Low, 1: Medium, 2: High

rules = tf.constant([
    # --- Kognitif ---
    [1, 0, 0, 0],  # rule 1
    [2, 0, 0, 0],
    [2, 0, 0, 1],
    [2, 0, 1, 0],
    [2, 0, 1, 1],
    [2, 1, 0, 0],
    [2, 1, 0, 1],
    [2, 1, 1, 0],
    [2, 1, 1, 1],
    # --- Afektif ---
    [0, 0, 0, 1],
    [0, 0, 0, 2],
    [0, 0, 1, 1],
    [0, 0, 1, 2],
    [0, 0, 2, 1],
    [0, 0, 2, 2],
    [0, 1, 0, 2],
    [0, 1, 1, 2],
    [0, 1, 2, 2],
    [1, 0, 0, 2],
    [1, 0, 1, 2],
    [1, 0, 2, 1],
    [1, 0, 2, 2],
    [1, 1, 0, 2],
    [1, 1, 1, 2],
    [1, 1, 2, 2],
    # --- Psikomotorik ---
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 2, 0],
    [0, 2, 0, 0],
    [0, 2, 0, 1],
    [0, 2, 1, 0],
    [0, 2, 1, 1],
    [0, 2, 2, 0],
    [0, 2, 2, 1],
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [1, 1, 2, 0],
    [1, 2, 0, 0],
    [1, 2, 0, 1],
    [1, 2, 1, 0],
    [1, 2, 1, 1],
    [1, 2, 2, 0],
    [1, 2, 2, 1],
    [2, 2, 0, 0],
    [2, 2, 0, 1],
    [2, 2, 1, 0],
    [2, 2, 1, 1],
    [2, 2, 2, 0],
    [2, 2, 2, 1],
], dtype=tf.int32)

# Konversi rules ke float dan normalisasi
rules_float = tf.cast(rules, tf.float32)
normalized_rules = tf.nn.l2_normalize(rules_float, axis=1)

# Pisahkan berdasarkan kelas
kognitif = normalized_rules[0:9]
afektif = normalized_rules[9:25]
psikomotorik = normalized_rules[25:49]

# Fungsi mencari rule dengan cosine similarity di atas threshold
def find_similar_rules_between(classA, classB, idx_offset_A, idx_offset_B, threshold=0.9):
    sim_matrix = tf.matmul(classA, classB, transpose_b=True).numpy()
    pairs = []
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            if sim_matrix[i, j] >= threshold:
                rule_A_index = idx_offset_A + i + 1
                rule_B_index = idx_offset_B + j + 1
                pairs.append((rule_A_index, rule_B_index, sim_matrix[i, j]))
    return pairs

# Kognitif - Psikomotorik
print("=== Kognitif - Psikomotorik ===")
similar_kp = find_similar_rules_between(kognitif, psikomotorik, 0, 25, threshold=0.85)
for r1, r2, sim in similar_kp:
    print(f"Rule {r1} (Kognitif) dan Rule {r2} (Psikomotorik) memiliki cosine similarity {sim:.4f}")
print()

# Kognitif - Afektif
print("=== Kognitif - Afektif ===")
similar_ka = find_similar_rules_between(kognitif, afektif, 0, 9, threshold=0.9)
for r1, r2, sim in similar_ka:
    print(f"Rule {r1} (Kognitif) dan Rule {r2} (Afektif) memiliki cosine similarity {sim:.4f}")
print()

# Afektif - Psikomotorik
print("=== Afektif - Psikomotorik ===")
similar_ap = find_similar_rules_between(afektif, psikomotorik, 9, 25, threshold=0.85)
for r1, r2, sim in similar_ap:
    print(f"Rule {r1} (Afektif) dan Rule {r2} (Psikomotorik) memiliki cosine similarity {sim:.4f}")
print()

# Simpan ke DataFrame
df_kp = pd.DataFrame(similar_kp, columns=["Rule Kognitif", "Rule Psikomotorik", "Cosine Similarity"])
df_ka = pd.DataFrame(similar_ka, columns=["Rule Kognitif", "Rule Afektif", "Cosine Similarity"])
df_ap = pd.DataFrame(similar_ap, columns=["Rule Afektif", "Rule Psikomotorik", "Cosine Similarity"])

# Bulatkan nilai cosine similarity ke 4 angka di belakang koma
df_kp["Cosine Similarity"] = df_kp["Cosine Similarity"].round(4)
df_ka["Cosine Similarity"] = df_ka["Cosine Similarity"].round(4)
df_ap["Cosine Similarity"] = df_ap["Cosine Similarity"].round(4)

# ====== SIMPAN HASIL COSINE SIMILARITY KE DALAM FILE EXCEL ======
with pd.ExcelWriter("static/28.cosine_similarity_results.xlsx") as writer:
    df_kp.to_excel(writer, sheet_name="Kognitif-Psikomotorik", index=False)
    df_ka.to_excel(writer, sheet_name="Kognitif-Afektif", index=False)
    df_ap.to_excel(writer, sheet_name="Afektif-Psikomotorik", index=False)

# Siapkan set untuk menyimpan rule yang akan dihapus
hapus_psikomotorik = set()
hapus_afektif = set()

# Ambil dari hasil cosine similarity
for r1, r2, sim in similar_kp:
    hapus_psikomotorik.add(r2)  # psikomotorik

for r1, r2, sim in similar_ap:
    hapus_afektif.add(r1)  # afektif

# Simpan hasil ke file .txt
with open("static/29.rules_tereliminasi.txt", "w") as f:
    f.write("Rule psikomotorik yang disarankan dihapus:\n")
    for r in sorted(hapus_psikomotorik):
        f.write(f"{r}\n")

    f.write("\nRule afektif yang disarankan dihapus:\n")
    for r in sorted(hapus_afektif):
        f.write(f"{r}\n")

# === REBUILD ANFIS MODEL ===
def build_anfis_model_with_excluded_rules(input_shape, excluded_rules=None):
    input_tensor = tf.keras.Input(shape=input_shape)  # Input 4 fitur

    # Layer 1: Fuzzifikasi
    fuzzy_output = tf.keras.layers.Lambda(fuzzy_layer_tf)(input_tensor)

    # Layer 2: Rule Layer dengan output_shape ditentukan
    output_dim = 49 if excluded_rules is None else 49 - len(excluded_rules)
    rule_output = tf.keras.layers.Lambda(
        lambda x: rule_layer(x, excluded_rules=excluded_rules),
        output_shape=(output_dim,)
    )(fuzzy_output)

    # Layer 3: Normalisasi (Softmax)
    T = 0.3
    normalized_rules = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x / T))(rule_output)

    # Layer 4 & 5: Defuzzifikasi
    rule_class_matrix = tf.gather(rule_classes, [i for i in range(49) if excluded_rules is None or i not in excluded_rules])
    output_layer = tf.keras.layers.Lambda(lambda x: tf.matmul(x, rule_class_matrix))(normalized_rules)

    model = tf.keras.Model(inputs=input_tensor, outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Penghapusan rules
excluded_rules_from_1 = [18, 24, 25, 35, 36, 41, 44, 45, 46, 47, 48, 49]  # index dari 1
excluded_rules = [i - 1 for i in excluded_rules_from_1]  # ubah ke index dari 0
anfis_model = build_anfis_model_with_excluded_rules((4,), excluded_rules=excluded_rules)
anfis_model.summary()

# RETRAIN MODEL
anfis_model.fit(X_train, Y_train, epochs=1, batch_size=2, verbose=1)

# Predict on test set
y_pred = anfis_model.predict(X_test)

# Ambil kelas dengan nilai tertinggi (one-hot encoding ke kategori)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = Y_test

# Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Plot Heatmap tanpa menampilkan plot, langsung simpan ke file
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1", "Class 2"], yticklabels=["Class 0", "Class 1", "Class 2"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
# Menyimpan gambar heatmap ke dalam folder static
plt.savefig('static/12. confusion_matrix_heatmap_RETRAIN.png')  # Menyimpan sebagai file gambar
plt.close()

# Hitung akurasi
accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f"Akurasi Model: {accuracy:.4f}")

# Tampilkan Classification Report
report = classification_report(y_true_classes, y_pred_classes, target_names=["Class 0", "Class 1", "Class 2"], zero_division=1)

# Mengonversi classification report menjadi DataFrame
from io import StringIO
import pandas as pd

# Menyimpan classification report dalam format yang bisa diimpor ke dalam DataFrame
report_str = classification_report(y_true_classes, y_pred_classes, target_names=["Class 0", "Class 1", "Class 2"], zero_division=1)

# ubah format report agar bisa dibaca oleh pandas
report_str = report_str.replace('precision', 'Precision').replace('recall', 'Recall').replace('f1-score', 'F1-Score').replace('support', 'Support')
report_str = report_str.replace('avg / total', 'Avg/Total').replace('macro avg', 'Macro Avg').replace('weighted avg', 'Weighted Avg')

# simpan report ke dalam file teks sementara
with open('static/13. classification_report_RETRAIN.txt', 'w') as f:
    f.write(report_str)  


# Convert report string to DataFrame using StringIO
report_df = pd.read_csv(StringIO(report_str), sep="\\s+", engine="python")

# Menyimpan report ke dalam file Excel
report_df.to_excel('static/13. classification_report_RETRAIN.xlsx', index=True)
print("\nClassification Report disimpan dalam file Excel di folder static.")

# Menghasilkan classification report
report = classification_report(y_true_classes, y_pred_classes, target_names=["Class 0", "Class 1", "Class 2"], zero_division=1)

# Membuat gambar dengan matplotlib
plt.figure(figsize=(6, 2))
plt.text(0.01, 1, report, {'fontsize': 12}, fontfamily='monospace', ha='left', va='top')
plt.axis('off')  # Menyembunyikan axis
plt.tight_layout()

# Simpan gambar sebagai file
plt.savefig("static/13. classification_report_RETRAIN.png", dpi=300)

# Buat DataFrame untuk melihat hasil klasifikasi
classification_results = pd.DataFrame(X_test_processed, columns=['Thinking', 'Striving', 'Relating', 'Influencing'])

# Tambahkan kolom kategori aktual dan prediksi
classification_results['Actual Class'] = y_true_classes
classification_results['Predicted Class'] = y_pred_classes

# Tampilkan 10 baris pertama hasil klasifikasi
print("\nHasil Klasifikasi (10 baris pertama):")
print(tabulate(classification_results.head(10), headers='keys', tablefmt='fancy_grid'))

# ====== SIMPAN HASIL KLASIFIKASI KE DALAM FILE EXCEL ======
classification_results.to_excel("static/14. classification_results_RETRAIN.xlsx", index=False)
print("Hasil klasifikasi disimpan dalam file Excel di folder static.")

# LABELLING

# Buat DataFrame dari X_test
classified_train_df = pd.DataFrame(X_test, columns=['Thinking', 'Striving', 'Relating', 'Influencing'])

# Pastikan y_true_classes dan y_pred_classes memiliki panjang yang sama dengan X_test
if len(y_true_classes) != len(X_test) or len(y_pred_classes) != len(X_test):
    raise ValueError("Ukuran y_true_classes atau y_pred_classes tidak sesuai dengan X_test")

# Tambahkan label asli dan hasil prediksi
classified_train_df['True_Label'] = y_true_classes
classified_train_df['Predicted_Label'] = y_pred_classes

# Mapping kategori dari angka ke label
label_mapping = {0: 'Kognitif', 1: 'Afektif', 2: 'Psikomotorik'}
classified_train_df['True_Label'] = classified_train_df['True_Label'].map(label_mapping)
classified_train_df['Predicted_Label'] = classified_train_df['Predicted_Label'].map(label_mapping)

# Tampilkan hasil dalam format tabel yang rapi
print(tabulate(classified_train_df.head(10), headers='keys', tablefmt='fancy_grid'))

# Pastikan folder 'static' ada
output_dir = "static"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Simpan DataFrame ke dalam file Excel
output_path = os.path.join(output_dir, "15. classified_train_df_RETRAIN.xlsx")
classified_train_df.to_excel(output_path, index=False)

print(f"Hasil klasifikasi disimpan dalam file Excel di folder {output_dir}.")

# Misal ini hasil perhitungan median
medians_per_rule = np.median(rule_output_val, axis=0)  # shape: (n_rules,)

# Buat label rule-nya
rule_names = [f"Rule {i+1}" for i in range(len(medians_per_rule))]

# Tampilkan hasil berpasangan
for name, median in zip(rule_names, medians_per_rule):
    print(f"{name}: {median:.4f}")

# Buat DataFrame untuk menyimpan hasil median
median_df = pd.DataFrame({
    'Rule': rule_names,
    'Median': medians_per_rule
})

# simpan median_df ke dalam file Excel
median_output_path = os.path.join(output_dir, "16. median_per_rule.xlsx")
median_df.to_excel(median_output_path, index=False)
print(f"Hasil median per rule disimpan dalam file Excel di folder {output_dir}.")

#load file xlsx dari static nama file "data_prediksi.xlsx"
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from skfuzzy import control as ctrl
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Pastikan folder 'static' ada
output_dir = "static"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# cek file data_prediksi.xlsx di folder static jika tidak ada, buat dataframe berisi default 48, 54, 54, 48
import os
if not os.path.exists(os.path.join(output_dir, "data_prediksi.xlsx")):  
    # Buat DataFrame default
    default_data = pd.DataFrame({
        'Thinking': [48],
        'Striving': [54],
        'Relating': [54],
        'Influencing': [48]
    })
    # Simpan ke dalam file Excel
    default_data.to_excel(os.path.join(output_dir, "data_prediksi.xlsx"), index=False)
    print(f"File data_prediksi.xlsx dibuat dengan nilai default di folder {output_dir}.")

# load dataframe dengan kolom 'Thinking', 'Striving', 'Relating', 'Influencing'
data_path = os.path.join(output_dir, "data_prediksi.xlsx")
processed_data = pd.read_excel(data_path)

# **Normalisasi Data**
scaler = MinMaxScaler(feature_range=(0, 1))  # Normalisasi 0-1
X_train_scaled = scaler.fit_transform(X_train_processed)  # Pastikan X_train_processed sudah ada dan dalam format yang benar
new_data_normalized = scaler.transform(processed_data.to_numpy())  # Konversi DataFrame ke numpy array

print(f"Data baru setelah normalisasi:\n{new_data_normalized}")

#simpan new_data_normalized ke dalam file excel
output_path = os.path.join(output_dir, "17. new_data_normalized.xlsx")
new_data_normalized_df = pd.DataFrame(new_data_normalized, columns=['Thinking', 'Striving', 'Relating', 'Influencing'])
new_data_normalized_df.to_excel(output_path, index=False)
print(f"Data setelah normalisasi disimpan dalam file Excel di folder {output_dir}.")



# **Prediksi menggunakan model ANFIS**
y_pred_new = anfis_model.predict(new_data_normalized)

# **Ambil kelas dengan nilai tertinggi**
predicted_class = np.argmax(y_pred_new, axis=1)

# Mapping kelas hasil prediksi
class_mapping = {0: 'Kognitif', 1: 'Afektif', 2: 'Psikomotorik'}

# **Tampilkan hasil prediksi**
print(f"Input Data:\n{processed_data}")
print(f"Data Setelah Normalisasi:\n{new_data_normalized}")
print(f"Hasil Prediksi (Class): {predicted_class[0]}")
print(f"Hasil Prediksi (Category): {class_mapping.get(predicted_class[0], 'Unknown')}")


# **Simpan hasil prediksi ke dalam file txt**
output_path = os.path.join(output_dir, "17. hasil_prediksi.txt")
with open(output_path, 'w') as file:
    file.write(f"Input Data:\n{processed_data}\n")
    file.write(f"Data Setelah Normalisasi:\n{new_data_normalized}\n")
    file.write(f"Hasil Prediksi (Class): {predicted_class[0]}\n")
    file.write(f"Hasil Prediksi (Category): {class_mapping.get(predicted_class[0], 'Unknown')}\n")

# Probabilitas pada setiap kelas tanpa softmax
for i, pred in enumerate(y_pred_new):
    print(f"Output tanpa softmax: {pred}")

# simpan output tanpa softmax ke dalam file txt
output_path = os.path.join(output_dir, "18. output_without_softmax.txt")
with open(output_path, 'w') as file:
    for i, pred in enumerate(y_pred_new):
        file.write(f"Output tanpa softmax: {pred}\n")

# Tampilkan hasil kemungkinan untuk setiap kelas dalam bentuk persentase
for i, pred in enumerate(y_pred_new):
    kognitif = pred[0] * 100
    afektif = pred[1] * 100
    psikomotorik = pred[2] * 100

    print(f"  Kognitif     : {kognitif:.2f}%")
    print(f"  Afektif      : {afektif:.2f}%")
    print(f"  Psikomotorik : {psikomotorik:.2f}%")

# simpan output tanpa softmax ke dalam file txt
output_path = os.path.join(output_dir, "19. output_probabilitas.txt")
with open(output_path, 'w') as file:
    for i, pred in enumerate(y_pred_new):
        kognitif = pred[0] * 100
        afektif = pred[1] * 100
        psikomotorik = pred[2] * 100

        file.write(f"  Kognitif     : {kognitif:.2f}%\n")
        file.write(f"  Afektif      : {afektif:.2f}%\n")
        file.write(f"  Psikomotorik : {psikomotorik:.2f}%\n")

# visualisasi hasil prediksi kognitif, afektif, psikomotorik
import matplotlib.pyplot as plt
import numpy as np

# Buat data baru dari pred[0] * 100, pred[1] * 100, pred[2] * 100
data = {
    'Kognitif': [pred[0] * 100 for pred in y_pred_new],
    'Afektif': [pred[1] * 100 for pred in y_pred_new],
    'Psikomotorik': [pred[2] * 100 for pred in y_pred_new]
}

# Buat DataFrame dari data baru
data_df = pd.DataFrame(data)

# Buat bar plot dari data baru
data_df.plot(kind='bar')
plt.xlabel('Kategori')
plt.ylabel('Persentase')
plt.title('Visualisasi Hasil Prediksi')
plt.savefig(os.path.join(output_dir, "20. visualisasi_hasil_prediksi.png"))

# GRAFIK NILAI BAKAT
# Buat bar plot dari data baru
data = processed_data.loc[0, ['Thinking', 'Striving', 'Relating', 'Influencing']]
values = data.values
labels = data.index
colors = ['#6A5ACD', '#FF69B4', '#20B2AA', '#FFA500']

n_bars = len(values)
bar_width = 0.15  # Lebar batang, kecil supaya ramping

# Total lebar semua batang tanpa jarak
total_bar_width = n_bars * bar_width

# Tentukan posisi x supaya bar-bar tepat di tengah [0,1]
# Tengahnya di 0.5, jadi posisi mulai:
start_x = 0.5 - total_bar_width / 2

# Buat posisi x masing-masing batang tanpa jarak antar batang
x_pos = [start_x + i * bar_width for i in range(n_bars)]

plt.figure(figsize=(6, 4))

bars = plt.bar(x_pos, values, width=bar_width, color=colors, align='edge')

# Tambah nilai di atas batang
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}%', 
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Label x di tengah tiap batang
plt.xticks(ticks=[x + bar_width/2 for x in x_pos], labels=labels)

# Set limit supaya grafik terlihat pas
plt.xlim(0, 1)
plt.ylim(0, 110)

plt.xlabel('Kategori')
plt.ylabel('Persentase')
plt.title('Persentase Bakat Siswa')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "30. visualisasi_nilai_bakat.png"))


# Menyimpan data variabel talents mapping
thinking_values = processed_data['Thinking'].tolist()
striving_values = processed_data['Striving'].tolist()
relating_values = processed_data['Relating'].tolist()
influencing_values = processed_data['Influencing'].tolist()

# Buat model baru dari input sampai output rule layer (tanpa softmax)
rule_layer_model = tf.keras.Model(inputs=anfis_model.input, outputs=anfis_model.get_layer(index=2).output)  # Layer ke-2 adalah rule layer

# Jalankan prediksi data baru untuk rule strengths
rule_strengths = rule_layer_model.predict(new_data_normalized)

# Tampilkan semua rule strengths untuk data baru
for i, strength in enumerate(rule_strengths[0]):
    print(f"Rule {i+1}: {strength:.4f}")

# Simpan rule strengths ke dalam file Excel
output_path = os.path.join(output_dir, "20. rule_strengths.xlsx")
pd.DataFrame(rule_strengths).to_excel(output_path, index=False)

# Firing Strength Per Rule

import matplotlib.pyplot as plt
import numpy as np

rule_labels = [f"Rule {i+1}" for i in range(len(rule_strengths[0]))]
strengths = rule_strengths[0]

plt.figure(figsize=(12, 6))
bars = plt.bar(rule_labels, strengths)
plt.title("")
plt.ylabel("Strength")
plt.xticks(rotation=45)
plt.tight_layout()

# Cari index nilai tertinggi
max_idx = np.argmax(strengths)
max_value = strengths[max_idx]

# Tambahkan label di atas batang tertinggi
plt.text(max_idx, max_value + 0.01, f"Max: {max_value:.3f}", ha='center', va='bottom', color='#de3b93', fontweight='bold')

# warnai batang tertinggi
bars[max_idx].set_color('#de3b93')

# Simpan plot ke dalam folder static
output_path = os.path.join(output_dir, "21. rule_strengths_plot.png")
plt.savefig(output_path)

plt.close() 

for i, pred in enumerate(y_pred_new):
    kognitif = pred[0] * 100
    afektif = pred[1] * 100
    psikomotorik = pred[2] * 100

    fig, ax = plt.subplots(figsize=(12, 1))

    warna_kognitif = '#586bdd'    # Biru tua
    warna_afektif = '#de3b93'     # Magenta
    warna_psikomotorik = '#fcb718' # Orange

    ax.barh(0, kognitif, color=warna_kognitif)
    ax.barh(0, afektif, left=kognitif, color=warna_afektif)
    ax.barh(0, psikomotorik, left=kognitif + afektif, color=warna_psikomotorik)

    # Hilangkan axis
    ax.axis('off')

    # Tambahkan teks di dalam bar
    ax.text(kognitif / 2, 0, f'Kognitif\n{kognitif:.2f}%', va='center', ha='center', color='white', fontsize=10, fontweight='bold')
    ax.text(kognitif + afektif / 2, 0, f'Afektif\n{afektif:.2f}%', va='center', ha='center', color='white', fontsize=10, fontweight='bold')
    ax.text(kognitif + afektif + psikomotorik / 2, 0, f'Psikomotorik\n{psikomotorik:.2f}%', va='center', ha='center', color='white', fontsize=10, fontweight='bold')

    plt.tight_layout()
    # simpan plot ke dalam folder static
    output_path = os.path.join(output_dir, f"22. hasil_prediksi.png")
    plt.savefig(output_path)
    plt.close()