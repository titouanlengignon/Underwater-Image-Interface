from flask import Flask, request, render_template, send_from_directory
import os

app = Flask(__name__)

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
    
    file = request.files['file']
    
    if file.filename == '':
        return "Aucun fichier sélectionné", 400
    
    if file and allowed_file(file.filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        return f"Fichier téléchargé avec succès : {file.filename}"

    return "Type de fichier non autorisé", 400

if __name__ == '__main__':
    app.run(debug=True, port=5501)
