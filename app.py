from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import sys
import glob
import requests
import time
import csv

app = Flask(__name__,)

UPLOAD_FOLDER = 'telechargements'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def send_images(host, port, image_dir):                                      
    filepaths = glob.glob(image_dir + '/*.*')
    filepaths = [fp.replace("\\", '/') for fp in filepaths if allowed_file(fp)]

    filenames = [[str(i), os.path.basename(fp)] for i, fp in enumerate(filepaths)]
    print("Fichiers envoyés :", filenames)
    
    # Étape 1 : Envoi des noms de fichiers
    url_filenames = f'http://{host}:{port}/filenames'
    response = requests.post(url_filenames, json=filenames, verify=False)

    # Étape 2 : Envoi des images
    files = [('image', (os.path.basename(fp), open(fp, 'rb'), 'image/jpeg')) for fp in filepaths]
    url_predict = f'http://{host}:{port}/predict'
    response = requests.post(url_predict, files=files, verify=False)

    if response.status_code != 200:
        raise Exception('Erreur lors de la prédiction')
    else:
        print('Prédiction réussie')
        return response.json()


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
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/list_images', methods=['GET'])
def list_images():
    """ Liste les images sauvegardées sur le serveur """
    image_files = os.listdir(app.config['UPLOAD_FOLDER'])
    image_files = [file for file in image_files if allowed_file(file)]  # Filtre les fichiers avec des extensions autorisées
    
    return jsonify({"saved_images": image_files}), 200




# SAVE RESULTS
@app.route('/save_results', methods=['POST'])
def save_results():
    print("Requête reçue pour /save_results")

    try:
        data = request.get_json()
        results = data.get("results", [])

        if not results:
            return jsonify({"error": "Aucun résultat à sauvegarder"}), 400

        timestamp = int(time.time())
        csv_filename = f"results_{timestamp}.csv"
        csv_file_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)

        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Fichier', 'Classe prédite', 'Indice de classe', 'Incertitude'])

            for res in results:
                pred = res["predictions"]["lithology"]
                writer.writerow([
                    res['filename'],
                    pred['predicted_class'],
                    pred['predicted_index'],
                    f"{pred['uncertainty']:.4f}"
                ])

        return jsonify({
            "message": f"Résultats sauvegardés dans {csv_filename}",
            "filename": csv_filename
        }), 200

    except Exception as e:
        return jsonify({"error": f"Erreur lors de la sauvegarde : {str(e)}"}), 500


# Inference

@app.route('/inference', methods=['POST'])
def inference():
    print("Envoi des images pour l'inférence...")
    inference_dict = send_images('172.17.0.3', 5500, app.config['UPLOAD_FOLDER'])
    print("Résultat reçu :", inference_dict)
    return jsonify(inference_dict)  # ⬅️ on retourne le vrai résultat



if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0", port=5500)
