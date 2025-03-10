#app.py

from flask import Flask, request, render_template
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

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "Aucun fichier trouvé"
        
        file = request.files['file']
        
        if file.filename == '':
            return "Aucun fichier sélectionné"
        
        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            return f"Fichier téléchargé avec succès : {file.filename}"

    return render_template('index.html')

if __name__ == '__main__':
    print("Application lancée")
    app.run(debug=True,port=5500)
