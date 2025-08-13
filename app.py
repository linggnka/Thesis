from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os

# Flask app setup
app = Flask(__name__)
app.secret_key = "supersecretkey"
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Blueprint imports
from routes.intro import intro_bp
from routes.dataset import dataset_bp
from routes.prediction import prediction_bp
from routes.model import model_bp


# Register blueprints
app.register_blueprint(intro_bp, url_prefix='/')
app.register_blueprint(dataset_bp, url_prefix='/dataset')
app.register_blueprint(model_bp, url_prefix='/model')
app.register_blueprint(prediction_bp, url_prefix='/prediction')

# Start the app
if __name__ == '__main__':
    app.run(debug=True)