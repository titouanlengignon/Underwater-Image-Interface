from flask import Flask, request, render_template, jsonify
import os
from app_service import app_service  # Import du blueprint

app = Flask(__name__,)

# Configurer le dossier de téléchargement
UPLOAD_FOLDER = 'telechargements'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Assurez-vous que le répertoire de téléchargement existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "Aucun fichier trouvé", 400
    
    files = request.files.getlist('file')
    uploaded_files = []
    
    for file in files:
        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            uploaded_files.append(file_path)

    # Retourner les chemins des fichiers téléchargés
    return jsonify(uploaded_files)

@app.route('/images')
def get_images():
    # Récupérer tous les fichiers dans le dossier UPLOAD_FOLDER
    images = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if allowed_file(filename):
            images.append(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return jsonify(images)

# Enregistre le blueprint pour l'interférence
app.register_blueprint(app_service)  # Cela permet d'ajouter les routes du blueprint à l'application principale

if __name__ == '__main__':
    app.run(debug=True, port=5000)


