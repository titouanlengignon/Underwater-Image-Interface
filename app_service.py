from flask import Blueprint, jsonify

# Crée un blueprint
app_service = Blueprint('app_service', __name__)

@app_service.route('/interference', methods=['GET'])
def interference():
    print("Ça marche")  # Affiche ce message dans la console si tout fonctionne
    return jsonify({"message": "Interférence exécutée avec succès !"})