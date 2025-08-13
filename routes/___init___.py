from flask import Flask
from routes.intro import intro_bp
from routes.dataset import dataset_bp
from routes.model import model_bp
from routes.prediction import prediction_bp

def create_app():
    app = Flask(__name__)

    @app.context_processor
    def inject_enumerate():
        return dict(enumerate=enumerate)
    

    # Register blueprints
    app.register_blueprint(intro_bp, url_prefix='/')
    app.register_blueprint(dataset_bp, url_prefix='/dataset')
    app.register_blueprint(model_bp, url_prefix='/model')
    app.register_blueprint(prediction_bp, url_prefix='/prediction')

    return app
