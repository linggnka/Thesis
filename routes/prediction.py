from flask import Blueprint, render_template, request, redirect, url_for, flash
import os
import subprocess
import pandas as pd
import re
import numpy as np
from models.cluster import proses_clustering
from flask import render_template


# Inisialisasi blueprint
prediction_bp = Blueprint('prediction', __name__)
UPLOAD_FOLDER = 'uploads/'
CLUSTER_IMAGE_PATH = 'static/clustering.png'
SILHOUETTE_PATH = 'static/silhouette.png'
CENTROID_PATH = 'static/centroids.xlsx'
SILHOUETTE_EXCEL_PATH = 'static/silhouette.xlsx'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

# Lokasi atau direktori tempat file berada
file_directory = "static/"  # Ganti dengan path yang sesuai

# deklarasi file classification.py
classification_path = 'classification.py'

# Fungsi untuk menulis status prediksi ke dalam file
def write_status_prediksi(status):
    file_path = 'static/status_prediksi.txt'  # Ganti dengan path file yang sesuai
    
    try:
        with open(file_path, 'w') as file:
            # Menulis status True atau False sebagai string ke dalam file
            file.write(str(status))  # Convert True/False menjadi string dan tulis ke file
        print(f"Status prediksi telah disimpan: {status}")
    except Exception as e:
        print(f"Terjadi kesalahan saat menulis ke file: {e}")

# Fungsi untuk membaca status prediksi dari file
def get_status_prediksi():
    file_path = 'static/status_prediksi.txt'  # Pastikan path file sesuai

    try:
        # Membuka file dan membaca isinya
        with open(file_path, 'r') as file:
            status = file.read().strip()  # Menghapus whitespace

            # Mengecek apakah status adalah True atau False
            if status.lower() == "true":
                return True  # Jika status adalah 'true' (case-insensitive)
            elif status.lower() == "false":
                return False  # Jika status adalah 'false' (case-insensitive)
            else:
                return False  # Jika tidak ada nilai yang sesuai, anggap False
    except FileNotFoundError:
        print(f"Error: File {file_path} tidak ditemukan.")
        return False  # Jika file tidak ada, anggap prediksi belum dilakukan
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
        return False  # Jika ada kesalahan lain, anggap prediksi belum dilakukan


@prediction_bp.route('/ ')
def prediction():
    # Here you can fetch all the necessary data and pass it to the template
    # load files
    hasil_prediksi = load_prediction_from_file()
    output_softmax = load_output_without_softmax()
    probabilities = load_probabilitas()
    rule_strengths = load_rule_strengths()

    return render_template('prediction.html', hasil_prediksi=hasil_prediksi, output_softmax=output_softmax, probabilities=probabilities, rule_strengths=rule_strengths)


# Global function to read files
def read_file(file_path):
    full_path = os.path.join(file_directory, file_path)  # Concatenate static path
    if not os.path.exists(full_path):
        print(f"Error: File {full_path} does not exist.")
        return None
    with open(full_path, 'r') as file:
        return file.read()


# Route untuk halaman utama prediksi
@prediction_bp.route('/')
def prediction_home():
    # Cek apakah prediksi sudah dilakukan berdasarkan status di file
    status_prediksi = get_status_prediksi()  # Baca status prediksi dari file

    # Load predictions from file
    hasil_prediksi = load_prediction_from_file()

    if hasil_prediksi is None:
        # If parsing failed, return an error message and stay on the same page
        return render_template('prediction.html', error_message="Error: Failed to load prediction data.")

    # Load other data (same as before)
    output_softmax = load_output_without_softmax()
    probabilities = load_probabilitas()
    rule_strengths = load_rule_strengths()

    # load data siswa
    data_siswa_path = 'static/1. data_siswa_prediksi.xlsx'
    data_siswa = pd.read_excel(data_siswa_path)

    # ganti nama kolom dengan nama, kelas, asalSekolah
    if 'Nama' in data_siswa.columns:
        data_siswa.rename(columns={'Nama': 'Nama', 'Kelas': 'Kelas', 'Asal Sekolah': 'asalSekolah'}, inplace=True)
    else:
        # Jika kolom tidak ada, buat DataFrame kosong
        data_siswa = pd.DataFrame(columns=['Nama', 'Kelas', 'asalSekolah'])

    # pada data_siswa, ambil saja nilai dan masukan kembali ke dalam dictionary, nama dictionarynya adalah data_siswa
    data_siswa = {
        'Nama': data_siswa.iloc[0]['Nama'],
        'Kelas': data_siswa.iloc[0]['Kelas'],
        'asalSekolah': data_siswa.iloc[0]['asalSekolah']
    }

    # Assuming that everything went well, render the template with the data
    return render_template('prediction.html',
                           hasil_prediksi=hasil_prediksi, 
                           output_softmax=output_softmax, 
                           probabilities=probabilities, 
                           rule_strengths=rule_strengths,
                           data_siswa=data_siswa,
                           prediksi=status_prediksi)


# Function to parse the input data from hasil_prediksi.txt
def load_prediction_from_file():
    content = read_file('17. hasil_prediksi.txt')
    if not content:
        return None

    # Initialize an empty dictionary to store the data
    normalized_data = []
    # Membaca file txt dan mengubahnya menjadi dictionary
    input_data = {}  # Pastikan input_data adalah dictionary

    # Pastikan path file sesuai dengan lokasi sebenarnya
    file_path = "static/1. slider_input.txt"  # Atau ganti dengan path absolut

    try:
        file = open(file_path, "r")
        
        for line in file:
            # Menghapus whitespace dan memisahkan berdasarkan ':'
            line = line.strip()
            if line:  # Cek jika baris tidak kosong
                try:
                    key, value = line.split(':')
                    input_data[key] = int(value)  # Coba konversi ke integer
                except ValueError:  # Menangani error jika nilai tidak bisa dikonversi ke integer
                    print(f"Warning: Baris '{line}' tidak valid, nilai tidak bisa diubah menjadi integer.")
                    continue  # Lewati baris yang tidak valid
        
        # Jangan lupa untuk menutup file setelah selesai
        file.close()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' tidak ditemukan.")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}") 

    # load data dari data_prediksi.xlsx
    data_prediksi_path = 'static/data_prediksi.xlsx'
    data_prediksi = pd.read_excel(data_prediksi_path)

    hasil_komposit = {}

    # Mengambil nilai dari DataFrame dengan variabel Thinking, Striving, Relating, Influencing
    hasil_komposit = {
        'Thinking': data_prediksi['Thinking'].values[0],
        'Striving': data_prediksi['Striving'].values[0],
        'Relating': data_prediksi['Relating'].values[0],
        'Influencing': data_prediksi['Influencing'].values[0]
    }




    

    # Parsing Normalized Data
    normalized_data_match = re.search(r"Data Setelah Normalisasi:\s*(.*?)\s*Hasil Prediksi", content, re.DOTALL)
    if normalized_data_match:
        normalized_data_section = normalized_data_match.group(1).strip()
        
        # Clean up the string and convert to a list of lists of floats
        normalized_data_section = normalized_data_section.replace("[[", "").replace("]]", "").replace("[", "").replace("]", "")
        
        # Split and process data
        data_split = normalized_data_section.split()
        # Check if the length of split data is divisible by 4
        if len(data_split) % 4 == 0:
            normalized_data = [
                [float(num) for num in data_split[i:i + 4]]
                for i in range(0, len(data_split), 4)
            ]
        else:
            print(f"Warning: Data length is not a multiple of 4. Data may be malformed. Length: {len(data_split)}")

    # Parsing Prediksi (Class and Category)
    predicted_class = None
    predicted_category = None
    class_match = re.search(r"Hasil Prediksi \(Class\):\s*(\d+)", content)
    if class_match:
        predicted_class = int(class_match.group(1))

    category_match = re.search(r"Hasil Prediksi \(Category\):\s*(\w+)", content)
    if category_match:
        predicted_category = category_match.group(1)

    # Remove duplicates from normalized_data
    normalized_data = [list(x) for x in set(tuple(x) for x in normalized_data)]

    # Return the parsed data as a dictionary
    return {
        "input_data": input_data,
        "normalized_data": normalized_data,
        "predicted_class": predicted_class,
        "predicted_category": predicted_category,
        "hasil_komposit": hasil_komposit
    }
# Function to load output without softmax
def load_output_without_softmax():
    
    #baca file di dalam folder static
    content = read_file('18. output_without_softmax.txt')
    if not content:
        return None

    match = re.search(r"Output tanpa softmax:\s*\[([^\]]+)\]", content)
    if match:
        array_str = match.group(1).strip()
        output_array = np.array([float(val) for val in array_str.split()])
        return output_array
    return None


# Function to load probabilitas
def load_probabilitas():
    content = read_file('19. output_probabilitas.txt')  # Ensure the correct file path
    if not content:
        return None

    probabilities = {}

    # Improved regex to capture category and probability
    matches = re.findall(r"(\w+[\w\s]*)\s*:\s*(\d+\.\d+)%", content)
    
    for match in matches:
        category = match[0].strip()  # Strip any unwanted spaces from the category
        probability = float(match[1])  # Convert string to float
        probabilities[category] = probability

    return probabilities

cek_probabilitas = load_probabilitas()
print(cek_probabilitas)


# Function to load rule strengths from an Excel file
def load_rule_strengths():
    file_path = 'static/20. rule_strengths.xlsx'
    content = read_file(file_path)
    if not content:
        print(f"Error: File {file_path} does not exist.")
        return []  # Return an empty list instead of None
    
    # Assuming you are using pandas to read the Excel file
    try:
        df = pd.read_excel(content, header=None)
        # Convert DataFrame to a list of lists
        rule_strengths = df.values.tolist()
        print("Data from rule_strengths.xlsx:")
        print(df)
        return rule_strengths
    except Exception as e:
        print(f"Error while reading Excel file: {e}")
        return []  # Return empty list on failure


# Fungsi untuk menghitung persentase
def hitung_persentase_bobot(total_nilai, max_nilai):
    # Menghitung persentase berdasarkan total nilai dan nilai maksimum kategori
    return total_nilai / max_nilai * 100 if max_nilai != 0 else 0


# Route untuk mengambil data dari form dan memprosesnya
# Fungsi untuk memproses data dan menyimpan ke file Excel
@prediction_bp.route('/get_data', methods=['POST'])
def get_data():


    # Ambil data dari form (nilai slider)
    sliders = {key: int(request.form.get(key)) for key in request.form if key.startswith('slider-')}
    
    # Breakdown sliders untuk kategori
    thinking = sum([sliders[f"slider-{i}"] for i in range(1, 17)])  # x1 - x16 (16 slider)
    striving = sum([sliders[f"slider-{i}"] for i in range(17, 35)]) # x17 - x34 (18 slider)
    relating = sum([sliders[f"slider-{i}"] for i in range(35, 53)]) # x35 - x52 (18 slider)
    influencing = sum([sliders[f"slider-{i}"] for i in range(53, 69)]) # x53 - x68 (16 slider)

    # simpan data input slider ke dalam txt
    with open('static/1. slider_input.txt', 'w') as f:
        f.write(f"Input Data:\n")
        f.write(f"Thinking: {thinking}\n")
        f.write(f"Striving: {striving}\n")
        f.write(f"Relating: {relating}\n")
        f.write(f"Influencing: {influencing}\n")

    
    # === Pattern Values untuk Normalisasi ===
    pattern_values = {
        "Thinking": 80,      # Maksimum nilai untuk Thinking
        "Striving": 90,      # Maksimum nilai untuk Striving
        "Relating": 90,      # Maksimum nilai untuk Relating
        "Influencing": 80    # Maksimum nilai untuk Influencing
    }

    # Ubah ke persentase berdasarkan pattern_values
    thinking_percentage = hitung_persentase_bobot(thinking, pattern_values["Thinking"])
    striving_percentage = hitung_persentase_bobot(striving, pattern_values["Striving"])
    relating_percentage = hitung_persentase_bobot(relating, pattern_values["Relating"])
    influencing_percentage = hitung_persentase_bobot(influencing, pattern_values["Influencing"])


    # Tentukan file path untuk data prediksi
    file_path = 'static/data_prediksi.xlsx'

    # Cek apakah file sudah ad
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
    else:
        # Buat DataFrame baru jika file tidak ada
        df = pd.DataFrame(columns=['Thinking', 'Striving', 'Relating', 'Influencing'])

    # Data baru untuk disimpan sebagai DataFrame
    new_data = {
        'Thinking': thinking_percentage,
        'Striving': striving_percentage,
        'Relating': relating_percentage,
        'Influencing': influencing_percentage
    }

    # Replace semua data di dalam DataFrame
    df = pd.DataFrame(columns=['Thinking', 'Striving', 'Relating', 'Influencing'])
    new_data_df = pd.DataFrame([new_data])

    # Check if the new data already exists in the DataFrame
    if not df[(df['Thinking'] == thinking) &
              (df['Striving'] == striving) &
              (df['Relating'] == relating) &
              (df['Influencing'] == influencing)].empty:
        # If the data exists, replace all rows with the new data, starting from row 1
        df = new_data_df  # Replace entire DataFrame with the new data
        flash("Data has been replaced and starts from row 1.", "info")
    else:
        # If data doesn't exist, replace all rows with the new data
        df = new_data_df  # Replace entire DataFrame with the new data
        flash("Data successfully saved!", "success")

    # Save the DataFrame to Excel after modification
    df.to_excel(file_path, index=False)

    # Tentukan file path untuk data siswa
    file_path_siswa = 'static/1. data_siswa_prediksi.xlsx'

    # Ambil data siswa dari form (misalnya, nama, kelas, asal sekolah)
    nama = request.form.get('nama')
    kelas = request.form.get('kelas')
    asalSekolah = request.form.get('asalSekolah')

    # Cek apakah file sudah ada
    if os.path.exists(file_path_siswa):
        df_siswa = pd.read_excel(file_path_siswa)
    else:
        # Buat DataFrame baru jika file tidak ada
        df_siswa = pd.DataFrame(columns=['Nama', 'Kelas', 'Asal Sekolah'])

    # Data baru untuk disimpan sebagai DataFrame
    siswa_data = {
        'Nama': nama,
        'Kelas': kelas,
        'Asal Sekolah': asalSekolah
    }

    # Replace semua data di dalam DataFrame
    siswa_data_df = pd.DataFrame([siswa_data])

    # Check if the new data already exists in the DataFrame
    if not df_siswa[(df_siswa['Nama'] == nama) &
                    (df_siswa['Kelas'] == kelas) &
                    (df_siswa['Asal Sekolah'] == asalSekolah)].empty:
        # If the data exists, replace all rows with the new data, starting from row 1
        df_siswa = siswa_data_df  # Replace entire DataFrame with the new data
        flash("Data has been replaced and starts from row 1.", "info")
    else:
        # If data doesn't exist, replace all rows with the new data
        df_siswa = siswa_data_df  # Replace entire DataFrame with the new data
        flash("Data successfully saved!", "success")

    # Save the DataFrame to Excel after modification
    df_siswa.to_excel(file_path_siswa, index=False)

    return redirect(url_for('prediction.prediction_home'))

# Routes untuk mengklasifikasi data
@prediction_bp.route('/classify_data', methods=['POST'])
def classify_data():
    # Cek file dataset
    files = os.listdir(UPLOAD_FOLDER)
    dataset_path = None
    dataset_name = None
    for file in files:
        if file.endswith('.xlsx') or file.endswith('.csv'):
            dataset_path = os.path.join(UPLOAD_FOLDER, file)
            dataset_name = file
            break

    if not dataset_path:
        flash('Dataset tidak ditemukan. Upload dulu.')
        return redirect(url_for('prediction.prediction_home'))

    # Jalankan clustering
    silhouette, centroid_df, membership_df, df_cluster_stats, labels, u = proses_clustering(
        dataset_path,
        save_path=CLUSTER_IMAGE_PATH,
        silhouette_path=SILHOUETTE_PATH,
        centroid_path=CENTROID_PATH
    )

    # Inisialisasi
    classification_path = 'models/classification.py'
    
    write_status_prediksi(True)  # Set status prediksi ke True

    # Jalankan ANFIS classification meskipun file sudah ada
    try:
        result = subprocess.run(['python', classification_path], capture_output=True, text=True)
        print("ANFIS stdout:", result.stdout)
        print("ANFIS stderr:", result.stderr)
    except Exception as e:
        flash(f"Gagal menjalankan ANFIS classification: {e}")
        return redirect(url_for('prediction.prediction_home'))

    return redirect(url_for('prediction.prediction_home'))
