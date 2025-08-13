from flask import Blueprint, render_template, request, redirect, url_for, flash
import os
import subprocess
import pandas as pd
from models.cluster import proses_clustering

model_bp = Blueprint('model', __name__)
UPLOAD_FOLDER = 'uploads/'
CLUSTER_IMAGE_PATH = 'static/clustering.png'
SILHOUETTE_PATH = 'static/silhouette.png'
CENTROID_PATH = 'static/centroids.xlsx'
SILHOUETTE_EXCEL_PATH = 'static/silhouette.xlsx'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

# Daftar nama file yang ingin diperiksa
file_list = [
    "12. confusion_matrix_heatmap_RETRAIN.png",
    "13. classification_report_RETRAIN.xlsx",
    "14. classification_results_RETRAIN.xlsx",
    "15. classified_train_df_RETRAIN.xlsx",
    "16. median_per_rule.xlsx"
]

# Lokasi atau direktori tempat file berada
file_directory = "static/"  # Ganti dengan path yang sesuai

# deklarasi file classification.py
classification_path = 'classification.py'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@model_bp.route('/', methods=['GET','POST'])
def model_home():
    clustering_exist = os.path.exists(CLUSTER_IMAGE_PATH)
    centroid_exist = os.path.exists(CENTROID_PATH)
    silhouette_exist = os.path.exists(SILHOUETTE_EXCEL_PATH)

    silhouette_score = None
    centroid_df = None
    membership_df = None
    checkModel = False
    confusion_matrix = {}

    # Baca Centroids dan Membership
    if centroid_exist:
        try:
            excel_data = pd.read_excel(CENTROID_PATH, sheet_name=None)
            centroid_df = excel_data.get('Centroids')
            membership_df = excel_data.get('Membership')
        except Exception as e:
            print("Gagal membaca centroids.xlsx:", e)

    # Baca Silhouette Score
    if silhouette_exist:
        try:
            silhouette_df = pd.read_excel(SILHOUETTE_EXCEL_PATH)
            if not silhouette_df.empty and 'Silhouette Score' in silhouette_df.columns:
                silhouette_score = silhouette_df['Silhouette Score'].iloc[0]
        except Exception as e:
            print("Gagal membaca silhouette.xlsx:", e)

    # Cek file lainnya
    confusion_image_exist = os.path.exists(os.path.join(file_directory, "12. confusion_matrix_heatmap_RETRAIN.png"))
    classification_result_exist = os.path.exists(os.path.join(file_directory, "14. classification_results_RETRAIN.xlsx"))
    classified_train_exist = os.path.exists(os.path.join(file_directory, "15. classified_train_df_RETRAIN.xlsx"))
    median_per_rule_exist = os.path.exists(os.path.join(file_directory, "16. median_per_rule.xlsx"))

    # Baca dataset
    files = os.listdir(UPLOAD_FOLDER)
    csv_files = [file for file in files if file.endswith('.csv')]
    xlsx_files = [file for file in files if file.endswith('.xlsx')]

    dataset_name = None
    dataset = None

    if csv_files:
        dataset_name = csv_files[0]
        dataset = pd.read_csv(os.path.join(UPLOAD_FOLDER, dataset_name))
    elif xlsx_files:
        dataset_name = xlsx_files[0]
        dataset = pd.read_excel(os.path.join(UPLOAD_FOLDER, dataset_name))

    return render_template('model_home.html',
                           clustering_exist=clustering_exist,
                           centroid_exist=centroid_exist,
                           silhouette_exist=silhouette_exist,
                           silhouette_score=silhouette_score,
                           centroid_df=centroid_df,
                           membership_df=membership_df,
                           dataset=dataset,
                           dataset_name=dataset_name,
                           checkModel=checkModel,
                           confusion_image_exist=confusion_image_exist,
                           classification_result_exist=classification_result_exist,
                           classified_train_exist=classified_train_exist,
                           median_per_rule_exist=median_per_rule_exist)

@model_bp.route('/train', methods=['POST'])
def train_model():
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
        return redirect(url_for('model.model_home'))

    # Jalankan clustering
    silhouette, centroid_df, membership_df, df_cluster_stats, labels, u = proses_clustering(
        dataset_path,
        save_path=CLUSTER_IMAGE_PATH,
        silhouette_path=SILHOUETTE_PATH,
        centroid_path=CENTROID_PATH
    )

    # Simpan centroid + membership ke satu file Excel
    with pd.ExcelWriter(CENTROID_PATH) as writer:
        centroid_df.to_excel(writer, sheet_name='Centroids', index=False)
        membership_df.to_excel(writer, sheet_name='Membership', index=False)

    # Simpan silhouette score
    silhouette_df = pd.DataFrame([{'Silhouette Score': silhouette}])
    silhouette_df.to_excel(SILHOUETTE_EXCEL_PATH, index=False)

    # Inisialisasi
    classification_path = 'models/classification.py'
    checkModel = False
    confusion_matrix = None
    dataset = None
    silhouette_score = None

    # Baca ulang centroid dan membership
    try:
        excel_data = pd.read_excel(CENTROID_PATH, sheet_name=None)
        centroid_df = excel_data.get('Centroids')
        membership_df = excel_data.get('Membership')
    except Exception as e:
        print("Gagal membaca centroids.xlsx:", e)

    # Baca silhouette score
    try:
        silhouette_df = pd.read_excel(SILHOUETTE_EXCEL_PATH)
        if not silhouette_df.empty and 'Silhouette Score' in silhouette_df.columns:
            silhouette_score = silhouette_df['Silhouette Score'].iloc[0]
    except Exception as e:
        print("Gagal membaca silhouette.xlsx:", e)

    # Baca dataset
    try:
        if dataset_path.endswith('.csv'):
            dataset = pd.read_csv(dataset_path)
        elif dataset_path.endswith('.xlsx'):
            dataset = pd.read_excel(dataset_path)
    except Exception as e:
        flash(f"Terjadi kesalahan saat membaca dataset: {e}")
        return redirect(url_for('model.model_home'))

    # Jalankan ANFIS classification meskipun file sudah ada
    try:
        result = subprocess.run(['python', classification_path], capture_output=True, text=True)
        print("ANFIS stdout:", result.stdout)
        print("ANFIS stderr:", result.stderr)
    except Exception as e:
        flash(f"Gagal menjalankan ANFIS classification: {e}")
        return redirect(url_for('model.model_home'))


    # Baca Hasil 14. classification_results_RETRAIN.xlsx dan jadikan tabel
    classification_results_path = os.path.join(file_directory, "14. classification_results_RETRAIN.xlsx")
    if os.path.exists(classification_results_path):
        checkModel = True
        try:
            classification_results_df = pd.read_excel(classification_results_path)
            classification_results_df.dropna(subset=['class'], inplace=True)
            classification_results_df = classification_results_df[['class', 'predicted_class', 'confidence']]
            classification_results_df.dropna(inplace=True)
        except Exception as e:
            classification_results_df = pd.DataFrame()
    else:
        flash('File classification_results_RETRAIN.xlsx tidak ditemukan.')
        classification_results_df = pd.DataFrame()

    # Baca dataset hasil klasifikasi 15. classified_train_df_RETRAIN.xlsx
    classified_train_path = os.path.join(file_directory, "15. classified_train_df_RETRAIN.xlsx")
    classified_train_name = None
    if os.path.exists(classified_train_path):
        try:
            classified_train_df = pd.read_excel(classified_train_path)
            classified_train_name = classified_train_path.split('/')[-1]
        except Exception as e:
            flash(f"Gagal membaca classified train df: {str(e)}")
            classified_train_df = pd.DataFrame()

    return render_template('model_home.html',
                           checkModel=checkModel,
                           dataset=dataset,
                           dataset_name=dataset_name,
                           clustering_exist=os.path.exists(CLUSTER_IMAGE_PATH),
                           centroid_df=centroid_df,
                           membership_df=membership_df,
                           classified_train_df=classified_train_df,
                           classified_train_df_name=classified_train_name,
                           silhouette_exist=os.path.exists(SILHOUETTE_EXCEL_PATH),
                           silhouette_score=silhouette_score)
