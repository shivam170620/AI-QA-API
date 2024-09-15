import os
from flask import Flask
from routes.doc_ingestion_route import doc_ingestion_route_blueprint

app = Flask(__name__)

# Setup upload folder and allowed file extensions
app.config['UPLOAD_FOLDER'] = './uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'md'}

app.register_blueprint(doc_ingestion_route_blueprint)

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Run the app
    app.run(port=5004)