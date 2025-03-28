from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import time

app = Flask(__name__)

UPLOAD_FOLDER = 'telechargements'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

# Supprimer la redondance ici, il faut conserver une seule route /upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier trouvé"}), 400
    
    files = request.files.getlist('file')
    uploaded_files = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Vérifier si le fichier existe déjà pour éviter l'écrasement
            if os.path.exists(file_path):
                base, ext = os.path.splitext(filename)
                counter = 1
                new_filename = f"{base}_{counter}{ext}"
                while os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], new_filename)):
                    counter += 1
                    new_filename = f"{base}_{counter}{ext}"
                filename = new_filename
            
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            uploaded_files.append(request.host_url + "uploads/" + filename)

    return jsonify(uploaded_files)

@app.route('/images')
def get_images():
    images = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if allowed_file(filename):
            images.append(request.host_url + "uploads/" + filename)
    return jsonify(images)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/list_images', methods=['GET'])
def list_images():
    """ Liste les images sauvegardées sur le serveur """
    image_files = os.listdir(app.config['UPLOAD_FOLDER'])
    image_files = [file for file in image_files if allowed_file(file)]  # Filtre les fichiers avec des extensions autorisées
    
    return jsonify({"saved_images": image_files}), 200

@app.route('/save_results', methods=['POST'])
def save_results():
    print("Requête reçue pour /save_results")  # Log pour vérifier si la requête arrive
    
    try:
        results = {
            "list": "Exemple de liste...",
            "results": "Exemple de résultat...",
            "category": "Exemple de catégorie...",
            "uncertainty": "Exemple d'incertitude..."
        }
        
        timestamp = int(time.time())
        txt_filename = f"results_{timestamp}.txt"
        txt_file_path = os.path.join(app.config['UPLOAD_FOLDER'], txt_filename)

        with open(txt_file_path, 'w') as f:
            f.write(f"📋 List: {results['list']}\n")
            f.write(f"\n🔍 Results: {results['results']}\n")
            f.write(f"\n🗂️ Category: {results['category']}\n")
            f.write(f"\n❓ Uncertainty: {results['uncertainty']}\n")

        return jsonify({
            "message": f"Fichier texte créé sous {txt_filename}"
        }), 200
    
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la création du fichier texte: {str(e)}"}), 500

@app.route('/inference', methods=['GET'])
def inference():
    result = {"message": "Inférence terminée avec succès !"}
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5500)
