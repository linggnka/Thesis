from flask import Blueprint, render_template

intro_bp = Blueprint('intro', __name__)

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





@intro_bp.route('/')
def intro():
    # nama aplikasi
    write_status_prediksi(False)
    return render_template('intro.html')


