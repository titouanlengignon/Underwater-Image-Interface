let imageFiles = []; // Stocke les fichiers sélectionnés
let currentIndex = 0; // Index de l'image affichée

const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const prevButton = document.getElementById("prevImage");
const nextButton = document.getElementById("nextImage");
const downloadBtn = document.getElementById("downloadBtn");
const interferenceBtn = document.getElementById("interference");

fileInput.addEventListener("change", function(event) {
    if (event.target.files.length > 0) {
        imageFiles = Array.from(event.target.files); // Stocker les images
        currentIndex = 0; // Reset à la première image
        displayImage();
        interferenceBtn.disabled = false; // Activer le bouton d'interférence
    }
});

function displayImage() {
    if (imageFiles.length > 0) {
        const file = imageFiles[currentIndex];
        const reader = new FileReader();

        reader.onload = function(e) {
            preview.src = e.target.result;
            preview.style.display = "block";
            downloadBtn.disabled = false; // Activer le bouton de téléchargement
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

// Télécharger l'image affichée
downloadBtn.addEventListener("click", function() {
    if (imageFiles.length > 0) {
        const file = imageFiles[currentIndex];
        const a = document.createElement("a");
        a.href = preview.src;
        a.download = file.name;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }
});

// Interférence (requête Flask)
if (interferenceBtn) {
    interferenceBtn.addEventListener("click", function() {
        if (imageFiles.length === 0) {
            alert("Veuillez d'abord sélectionner une image !");
            return;
        }

        fetch('/interference', { method: 'GET' })
        .then(response => response.json())
        .then(data => alert(data.message))  // Affiche la réponse en alerte
        .catch(error => console.error('Erreur:', error));
    });
} else {
    console.error("Bouton 'Interference' non trouvé !");
}
