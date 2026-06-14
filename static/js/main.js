// ============================================================
// Tabs
// ============================================================
const tabButtons = document.querySelectorAll(".tab-btn");
const tabContents = document.querySelectorAll(".tab-content");

tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
        tabButtons.forEach(b => b.classList.remove("active"));
        tabContents.forEach(c => c.classList.remove("active"));

        btn.classList.add("active");
        document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");

        if (btn.dataset.tab === "train") {
            loadDatasetInfo();
        }
    });
});

// ============================================================
// Detect tab elements
// ============================================================
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const previewSection = document.getElementById("previewSection");
const previewImg = document.getElementById("previewImg");
const removeBtn = document.getElementById("removeBtn");
const detectBtn = document.getElementById("detectBtn");
const loading = document.getElementById("loading");
const results = document.getElementById("results");
const errorBox = document.getElementById("errorBox");
const modelStatusText = document.getElementById("modelStatusText");
const modelStatus = document.getElementById("modelStatus");

let selectedFile = null;

// File input click
dropZone.addEventListener("click", () => fileInput.click());

// File selected
fileInput.addEventListener("change", (e) => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
});

// Drag & Drop
dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragging");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragging");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragging");
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith("image/")) {
        showError("Only image files are allowed!");
        return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        dropZone.style.display = "none";
        previewSection.style.display = "block";
        hideError();
        hideResults();
    };
    reader.readAsDataURL(file);
}

// Remove image
removeBtn.addEventListener("click", () => {
    selectedFile = null;
    fileInput.value = "";
    previewSection.style.display = "none";
    dropZone.style.display = "block";
    hideResults();
    hideError();
});

// Detect button
detectBtn.addEventListener("click", () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("image", selectedFile);

    loading.style.display = "block";
    hideResults();
    hideError();
    detectBtn.disabled = true;
    detectBtn.textContent = "⏳ Detecting...";

    fetch("/detect", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        loading.style.display = "none";
        detectBtn.disabled = false;
        detectBtn.textContent = "🔍 Detect";

        if (!data.success) {
            showError(data.error || "Something went wrong!");
            return;
        }

        showResults(data);
    })
    .catch(() => {
        loading.style.display = "none";
        detectBtn.disabled = false;
        detectBtn.textContent = "🔍 Detect";
        showError("Could not connect to the server. Is the Flask app running?");
    });
});

// Try Again
document.getElementById("tryAgainBtn").addEventListener("click", () => {
    removeBtn.click();
});

function showResults(data) {
    document.getElementById("resultEmoji").textContent = data.emoji || "🐾";
    document.getElementById("resultLabel").textContent = data.top_prediction;
    document.getElementById("confValue").textContent = data.confidence;

    const fill = document.getElementById("confFill");
    fill.style.width = "0%";
    setTimeout(() => {
        fill.style.width = `${data.confidence}%`;
    }, 100);

    // Predictions list
    const list = document.getElementById("predictionsList");
    list.innerHTML = "";
    data.all_predictions.forEach((pred, i) => {
        const item = document.createElement("div");
        item.className = "prediction-item";
        item.innerHTML = `
            <span class="pred-name">${i + 1}. ${pred.label}</span>
            <div class="pred-bar-wrapper">
                <div class="pred-bar">
                    <div class="pred-fill" style="width: ${pred.confidence}%"></div>
                </div>
            </div>
            <span class="pred-conf">${pred.confidence}%</span>
        `;
        list.appendChild(item);
    });

    results.style.display = "block";
}

function showError(msg) {
    document.getElementById("errorMsg").textContent = "⚠️ " + msg;
    errorBox.style.display = "block";
}

function hideError() { errorBox.style.display = "none"; }
function hideResults() { results.style.display = "none"; }

// ============================================================
// Train tab
// ============================================================
const datasetSummary = document.getElementById("datasetSummary");
const refreshDatasetBtn = document.getElementById("refreshDatasetBtn");
const trainBtn = document.getElementById("trainBtn");
const epochsInput = document.getElementById("epochsInput");
const batchSizeInput = document.getElementById("batchSizeInput");
const trainProgress = document.getElementById("trainProgress");
const progressFill = document.getElementById("progressFill");
const trainMessage = document.getElementById("trainMessage");
const trainErrorBox = document.getElementById("trainErrorBox");
const trainErrorMsg = document.getElementById("trainErrorMsg");
const trainSuccessBox = document.getElementById("trainSuccessBox");
const trainSuccessMsg = document.getElementById("trainSuccessMsg");

let pollTimer = null;

function loadDatasetInfo() {
    fetch("/dataset-info")
        .then(res => res.json())
        .then(data => {
            renderDatasetInfo(data);
            updateModelStatus(data.model_trained);
        })
        .catch(() => {
            datasetSummary.innerHTML = `<p class="muted">Could not load dataset info.</p>`;
        });
}

function renderDatasetInfo(data) {
    if (!data.classes || data.classes.length === 0) {
        datasetSummary.innerHTML = `
            <p class="muted">No labeled images found yet.</p>
            <p class="muted">Add photos to <code>dataset/images/</code> (with filenames like
            <code>cat_0001.jpg</code>, or with a <code>dataset/labels.csv</code> file).</p>
        `;
        trainBtn.disabled = true;
        return;
    }

    let html = `<p><strong>${data.classes.length}</strong> classes, <strong>${data.total_images}</strong> total images</p>`;
    html += `<div class="class-list">`;
    data.classes.forEach(c => {
        html += `<div class="class-item"><span>${c.name}</span><span class="class-count">${c.count} imgs</span></div>`;
    });
    html += `</div>`;

    if (data.classes.length < 2) {
        html += `<p class="muted">⚠️ Need at least 2 classes to train.</p>`;
        trainBtn.disabled = true;
    } else {
        trainBtn.disabled = false;
    }

    datasetSummary.innerHTML = html;
}

function updateModelStatus(modelTrained) {
    if (modelTrained) {
        modelStatusText.textContent = "✅ A trained model is ready - you can detect images below.";
        modelStatus.classList.remove("status-warning");
        modelStatus.classList.add("status-ok");
    } else {
        modelStatusText.textContent = "⚠️ No trained model yet. Go to the 'Train Model' tab to train one.";
        modelStatus.classList.remove("status-ok");
        modelStatus.classList.add("status-warning");
    }
}

refreshDatasetBtn.addEventListener("click", loadDatasetInfo);

trainBtn.addEventListener("click", () => {
    const epochs = parseInt(epochsInput.value) || 10;
    const batchSize = parseInt(batchSizeInput.value) || 16;

    trainErrorBox.style.display = "none";
    trainSuccessBox.style.display = "none";

    fetch("/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ epochs, batch_size: batchSize })
    })
    .then(res => res.json().then(data => ({ ok: res.ok, data })))
    .then(({ ok, data }) => {
        if (!ok) {
            trainErrorMsg.textContent = "⚠️ " + data.error;
            trainErrorBox.style.display = "block";
            return;
        }
        startPolling();
    })
    .catch(() => {
        trainErrorMsg.textContent = "⚠️ Could not connect to the server.";
        trainErrorBox.style.display = "block";
    });
});

function startPolling() {
    trainBtn.disabled = true;
    trainProgress.style.display = "block";
    progressFill.style.width = "0%";
    trainMessage.textContent = "Starting...";

    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(pollTrainStatus, 1500);
    pollTrainStatus();
}

function pollTrainStatus() {
    fetch("/train-status")
        .then(res => res.json())
        .then(data => {
            progressFill.style.width = `${data.percent}%`;
            trainMessage.textContent = data.message || "";

            if (data.error) {
                clearInterval(pollTimer);
                pollTimer = null;
                trainBtn.disabled = false;
                trainErrorMsg.textContent = "⚠️ " + data.error;
                trainErrorBox.style.display = "block";
                return;
            }

            if (data.done) {
                clearInterval(pollTimer);
                pollTimer = null;
                trainBtn.disabled = false;

                const acc = data.result ? (data.result.best_val_acc * 100).toFixed(1) : "?";
                trainSuccessMsg.textContent =
                    `🎉 Training complete! Best validation accuracy: ${acc}%. ` +
                    `Switch to the Detect tab to try it out.`;
                trainSuccessBox.style.display = "block";

                updateModelStatus(true);
            }
        })
        .catch(() => {
            // keep polling even if a single request fails
        });
}

// ============================================================
// Init
// ============================================================
loadDatasetInfo();
