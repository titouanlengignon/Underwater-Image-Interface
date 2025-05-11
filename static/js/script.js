// Déclaration des variables

let imageFiles = []; // Stocke les fichiers sélectionnés
let currentIndex = 0; // Index de l'image affichée

const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const prevButton = document.getElementById("prevImage");
const nextButton = document.getElementById("nextImage");
const downloadBtn = document.getElementById("downloadBtn");
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

        displayPredictionForCurrentImage(imageFiles[currentIndex].name);

    }
});

function displayPredictionForCurrentImage(filename) {
    const container = document.getElementById("results-container");
    container.innerHTML = ""; // Nettoyage avant affichage

    if (!window.lastInferenceResults) return;

    const result = window.lastInferenceResults.find(res => res.filename === filename);
    const criterion = document.getElementById("criterion-select").value;

    if (result && result.predictions[criterion]) {
        const prediction = result.predictions[criterion];
        container.innerHTML = `
            <div class="prediction-card">
                <h3> 📋 Name : ${result.filename}</h3>
                <p><strong> 🗂️ Category :</strong> ${prediction.predicted_class}</p>
                <p><strong> ❓ Uncertainty :</strong> ${prediction.uncertainty.toFixed(3)}</p>
            </div>
        `;
    } else {
        container.innerHTML = `
            <div class="prediction-card">
                <p>Aucun résultat pour cette image.</p>
            </div>
        `;
    }
}



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
        displayPredictionForCurrentImage(imageFiles[currentIndex].name);

    }
});

nextButton.addEventListener("click", function() {
    if (imageFiles.length > 0 && currentIndex < imageFiles.length - 1) {
        currentIndex++;
        displayImage();
        displayPredictionForCurrentImage(imageFiles[currentIndex].name);

    }
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
            alert("Images have been saved on the server!");
            console.log(" Saved files : ", data);
        } else {
            alert("Error during file upload");
        }
    })
    .catch(error => {
        console.error("Upload error:", error);
    });
});

// Bouton download en csv
downloadBtn.addEventListener("click", function () {
    if (!window.lastInferenceResults) {
        alert("No results to save!");
        return;
    }

    fetch('/save_results', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ results: window.lastInferenceResults })
    })
    .then(response => response.json())
    .then(data => {
        if (data.filename) {
            // Création du lien et téléchargement automatique
            const link = document.createElement("a");
            link.href = `/uploads/${data.filename}`;
            link.download = data.filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            alert("Results downloaded successfully!");
        } else {
            alert("File generated but filename is missing.");
        }
    })
    .catch(error => {
        console.error("Error during saving :", error);
        alert("Error during saving.");
    });
});



// Effets sur le bouton download
downloadBtn.addEventListener("mouseenter", () => {
    downloadBtn.innerHTML = "📥 Download";
});

downloadBtn.addEventListener("mouseleave", () => {
    downloadBtn.innerHTML = "⬇ Download";
});

// Inférence (requête Flask)
inferenceBtn.addEventListener("click", function () {
    const criterion = document.getElementById("criterion-select").value;
    const formData = new FormData();
    formData.append("criterion", criterion);

    // 👉 Afficher le loader
    const loader = document.getElementById("loader");
    if (loader) loader.style.display = "block";

    fetch('/inference', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // ❌ Cacher le loader une fois terminé
        if (loader) loader.style.display = "none";

        const container = document.getElementById("results-container");
        container.innerHTML = ''; // Clear previous results

        if (data.results && data.results.length > 0) {
            const criterion = document.getElementById("criterion-select").value;

            data.results.forEach(result => {
                const pred = result.predictions[criterion];
                const div = document.createElement("div");
                div.className = "result-block";

                if (pred) {
                    div.innerHTML = `
                        <p> 📋 Name : ${result.filename}</p>
                        <p> 🗂️ Category : <strong>${pred.predicted_class}</strong></p>
                        <p> ❓ Uncertainty : ${pred.uncertainty.toFixed(3)}</p>
                        <hr>
                    `;
                } else {
                    div.innerHTML = `
                        <p> 📋 Name : ${result.filename}</p>
                        <p> ⚠️ No results for the selected criterion (${criterion})</p>
                        <hr>
                    `;
                }

                container.appendChild(div);
            });

            document.getElementById("criterion-select").addEventListener("change", () => {
                displayPredictionForCurrentImage(imageFiles[currentIndex]?.name);
            });

            window.lastInferenceResults = data.results;
            displayPredictionForCurrentImage(imageFiles[currentIndex].name);
        } else {
            container.innerHTML = "<p>No predictions found</p>";
        }
    })
    .catch(error => {
        if (loader) loader.style.display = "none"; // ❌ Cacher même en cas d'erreur
        console.error('Erreur:', error);
    });
});
