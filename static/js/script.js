// Déclaration des variables

let imageFiles = []; // Stocke les fichiers sélectionnés
let currentIndex = 0; // Index de l'image affichée

const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const prevButton = document.getElementById("prevImage");
const nextButton = document.getElementById("nextImage");
const downloadBtn = document.getElementById("downloadBtn");
const inference = document.getElementById("inference");


console.log("Script chargé !");


// Evenement : séléction d'images
fileInput.addEventListener("change", function(event) {
    if (event.target.files.length > 0) {
        imageFiles = Array.from(event.target.files); // Stocker les images
        currentIndex = 0; // Reset à la première image
        displayImage();
        interference.disabled = false; // Activer le bouton d'interférence
    }
});

// Affichage de l'image
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

// Effets sur le bouton download
downloadBtn.addEventListener("mouseenter", () => {
    downloadBtn.innerHTML = "📥 Download";
});

downloadBtn.addEventListener("mouseleave", () => {
    downloadBtn.innerHTML = "⬇ Download";
});

// Inférence (requête Flask)
if (inference) {
    inference.addEventListener("click", function() {
        if (imageFiles.length === 0) {
            alert("Veuillez d'abord sélectionner une image !");
            return;
        }

        fetch('/inference', { method: 'GET' })
        .then(response => response.json())
        .then(data => alert(data.message))  // Affiche la réponse en alerte
        .catch(error => console.error('Erreur:', error));
    });
} else {
    console.error("Bouton 'Inference' non trouvé !");
}


// footer
// const footer = document.getElementById("footer");

// footer.classList.add("footer-visible");


// window.addEventListener("scroll", () => {
//     const scrollPosition = window.innerHeight + window.scrollY;
//     const pageHeight = document.documentElement.scrollHeight;

//     console.log(`Scroll Position: ${scrollPosition}`);
//     console.log(`Page Height: ${pageHeight}`);

//     if (scrollPosition >= pageHeight - 10) { 
//         console.log("Ajout de la classe footer-visible");
//         footer.classList.add("footer-visible"); // Fait apparaître le footer
//     } else {
//         console.log("Suppression de la classe footer-visible");
//         footer.classList.remove("footer-visible"); // Le cache quand on remonte
//     }
// });



