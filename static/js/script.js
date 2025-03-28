// Déclaration des variables
let imageFiles = []; // Stocke les fichiers sélectionnés
let currentIndex = 0; // Index de l'image affichée

const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const prevButton = document.getElementById("prevImage");
const nextButton = document.getElementById("nextImage");
const downloadBtn = document.getElementById("downloadBtn");
const saveBtn = document.getElementById("saveBtn");
const inferenceBtn = document.getElementById("inference"); 

console.log("Script chargé !");

// Événement : sélection d'images
fileInput.addEventListener("change", function(event) {
    if (event.target.files.length > 0) {
        imageFiles = Array.from(event.target.files); // Stocker les images
        currentIndex = 0; // Reset à la première image
        displayImage();
        if (inferenceBtn) {
            inferenceBtn.disabled = false; // Activer le bouton d'inférence si l'élément existe
        }
        saveBtn.disabled = false; // Activer le bouton "Save on Server"
    }
});

// Affichage de l'image sélectionnée
function displayImage() {
    if (imageFiles.length > 0) {
        const file = imageFiles[currentIndex];
        const reader = new FileReader();

        reader.onload = function(e) {
            preview.src = e.target.result;
            preview.style.display = "block";
            downloadBtn.disabled = false; // Activer le bouton "Download"
        };

        reader.readAsDataURL(file);
    }
}

// Navigation entre les images
prevButton.addEventListener("click", function() {
    if (imageFiles.length > 0 && currentIndex > 0) {
        currentIndex--;
        displayImage();
    }
});

nextButton.addEventListener("click", function() {
    if (imageFiles.length > 0 && currentIndex < imageFiles.length - 1) {
        currentIndex++;
        displayImage();
    }
});

// Inférence (requête Flask)
inferenceBtn.addEventListener("click", function() {
    fetch('/inference', { method: 'GET' })
    .then(response => response.json())  // Assurez-vous que la réponse est au format JSON
    .then(data => alert(data.message))  // Affiche le message de la réponse JSON
    .catch(error => console.error('Erreur:', error));
});





// Fonction pour sauvegarder les images sur le serveur
saveBtn.addEventListener("click", function() {
    const formData = new FormData();

    // Ajouter chaque image sélectionnée au FormData
    imageFiles.forEach((file, index) => {
        formData.append('file', file);
    });

    // Envoi des fichiers vers le serveur
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.length > 0) {
            // Affiche un message ou une action après l'upload réussi
            alert("Les images ont été sauvegardées sur le serveur !");
            console.log("Fichiers sauvegardés : ", data);
        } else {
            alert("Erreur lors de l'upload des fichiers.");
        }
    })
    .catch(error => {
        console.error('Erreur lors de l\'upload:', error);
    });
});

// Téléchargement du fichier texte 
downloadBtn.addEventListener("click", function() {
    const list = document.querySelector(".text-result-List").innerText || "Aucune donnée";
    const results = document.querySelector(".text-result-Results").innerText || "Aucune donnée";
    const category = document.querySelector(".text-result-Category").innerText || "Aucune donnée";
    const uncertainty = document.querySelector(".text-result-Uncertainty").innerText || "Aucune donnée";

    const fileContent = 
        "📋 List : " + list + "\n" +
        "🔍 Results : " + results + "\n" +
        "🗂️ Category : " + category + "\n" +
        "❓ Uncertainty : " + uncertainty;

    const blob = new Blob([fileContent], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "image_results.txt";

    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
});

// Effets sur le bouton download
downloadBtn.addEventListener("mouseenter", () => {
    downloadBtn.innerHTML = "📥 Download";
});

downloadBtn.addEventListener("mouseleave", () => {
    downloadBtn.innerHTML = "⬇ Download";
});
