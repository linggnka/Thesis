from flask import Blueprint, render_template, request, redirect, url_for, flash, send_file
import os
import pandas as pd
from io import BytesIO

# Inisialisasi blueprint
dataset_bp = Blueprint('dataset', __name__)

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'xlsx'}
MAIN_DATASET_PATH = os.path.join(UPLOAD_FOLDER, 'main_dataset.txt')

# Pastikan folder upload ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Mengecek dan menampilkan dataset utama
@dataset_bp.route('/cek', methods=['GET', 'POST'])
def cek_dataset():
    if not os.path.exists(MAIN_DATASET_PATH):
        flash('Dataset utama belum ditentukan. Silakan unggah terlebih dahulu.')
        return redirect(url_for('intro.intro'))

    with open(MAIN_DATASET_PATH, 'r') as f:
        dataset_name = f.read().strip()

    filepath = os.path.join(UPLOAD_FOLDER, dataset_name)

    if not os.path.exists(filepath):
        flash('Dataset utama tidak ditemukan di folder.')
        return redirect(url_for('intro.intro'))

    dataset = pd.read_excel(filepath)
    return render_template(
        'model_home.html',
        dataset=dataset,
        dataset_name=dataset_name,
        clustering_exist=False,
        centroid_exist=False,
        silhouette_exist=False,
        silhouette_score=None,
        centroid_df=None,
        membership_df=None
    )


# Mengunggah dataset dan menjadikannya dataset utama
@dataset_bp.route('/input', methods=['GET', 'POST'])
def input_dataset():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Tidak ada file di request.')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('Tidak ada file yang dipilih.')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Simpan nama file sebagai dataset utama
            with open(MAIN_DATASET_PATH, 'w') as f:
                f.write(filename)

            dataset = pd.read_excel(filepath)

            dataset_name = filename
            flash(f'Dataset {filename} berhasil diunggah dan ditandai sebagai dataset utama.')
            return render_template(
                'model_home.html',
                dataset=dataset,
                dataset_name=dataset_name,
                clustering_exist=False,
                centroid_exist=False,
                silhouette_exist=False,
                silhouette_score=None,
                centroid_df=None,
                membership_df=None
            )


        flash('Format file tidak diizinkan. Hanya file .xlsx diperbolehkan.')
        return redirect(request.url)

    return redirect(url_for('dataset.cek_dataset'))

# Simulasi pembuatan model
@dataset_bp.route('/model', methods=['GET', 'POST'])
def model_creation():
    if request.method == 'POST':
        # Proses pelatihan model bisa ditambahkan di sini
        flash('Model berhasil dibuat (simulasi).')
        return redirect(url_for('dataset.save_info'))

    return redirect(url_for('dataset.cek_dataset'))

# Menyimpan hasil model atau dataset
@dataset_bp.route('/save', methods=['GET', 'POST'])
def save_info():
    if not os.path.exists(MAIN_DATASET_PATH):
        flash('Dataset utama belum tersedia.')
        return redirect(url_for('dataset.cek_dataset'))

    with open(MAIN_DATASET_PATH, 'r') as f:
        dataset_name = f.read().strip()

    filepath = os.path.join(UPLOAD_FOLDER, dataset_name)

    if not os.path.exists(filepath):
        flash('Dataset utama tidak ditemukan.')
        return redirect(url_for('dataset.cek_dataset'))

    table_data = pd.read_excel(filepath).head()

    if request.method == 'POST':
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            table_data.to_excel(writer, index=False, sheet_name='Sheet1')
        output.seek(0)
        return send_file(output, as_attachment=True,
                         download_name=dataset_name,
                         mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    return render_template('save_info.html', table_data=table_data)
