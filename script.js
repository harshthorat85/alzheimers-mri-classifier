// ── API Configuration ────────────────────────────────────────────────────────
const API_URL = "https://harshthorat85-alzheimers-mri-classifier.hf.space/predict";

// ── DOM Elements ─────────────────────────────────────────────────────────────
const dropZone        = document.getElementById('dropZone');
const fileInput       = document.getElementById('fileInput');
const previewImage    = document.getElementById('previewImage');
const previewWrapper  = document.getElementById('previewWrapper');
const previewFilename = document.getElementById('previewFilename');
const resultsContainer = document.getElementById('resultsContainer');
const warningBanner   = document.getElementById('warningBanner');

// ── Drag and Drop ─────────────────────────────────────────────────────────────
['dragenter', 'dragover'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    }, false);
});

['dragleave', 'drop'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
    }, false);
});

dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) handleImageUpload(files[0]);
});

fileInput.addEventListener('change', (e) => {
    if (fileInput.files.length > 0) handleImageUpload(fileInput.files[0]);
});

// ── Core Upload Handler ───────────────────────────────────────────────────────
async function handleImageUpload(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload a valid image file (JPEG, PNG, or TIFF).');
        return;
    }

    // Show image preview and filename
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onloadend = () => {
        previewImage.src = reader.result;
        previewWrapper.style.display  = 'flex';
        dropZone.style.display        = 'none';
        previewFilename.textContent   = file.name;
    };

    // Prepare payload
    const formData = new FormData();
    formData.append('file', file);

    // Reset + loading state
    warningBanner.style.display = 'none';
    resultsContainer.innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
            Analyzing tissue density patterns…
        </div>`;

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        renderResults(data);

    } catch (error) {
        console.error('API Error:', error);
        resultsContainer.innerHTML = `
            <div class="error">
                ❌ Could not reach the inference server.
                <small>${error.message}</small>
            </div>`;
    }
}

// ── Render Results ────────────────────────────────────────────────────────────
function renderResults(data) {
    resultsContainer.innerHTML = '';

    if (data.low_confidence_warning) {
        warningBanner.innerText = "⚠️ Low Confidence: This image's signature deviates from the expected training distribution. Results may be unreliable.";
        warningBanner.style.display = 'block';
    }

    const list = document.createElement('div');
    list.className = 'results-list';

    Object.entries(data.confidence_scores).forEach(([className, score]) => {
        const percentage   = (score * 100).toFixed(1);
        const isTop        = className === data.prediction;

        const barRow = document.createElement('div');
        barRow.className = `bar-row ${isTop ? 'winner' : ''}`;
        barRow.innerHTML = `
            <div class="bar-labels">
                <span class="class-name">
                    ${className}
                    ${isTop ? '<span class="prediction-badge">top prediction</span>' : ''}
                </span>
                <span class="percentage-val">${percentage}%</span>
            </div>
            <div class="bar-track">
                <div class="bar-fill" data-width="${percentage}"></div>
            </div>
        `;
        list.appendChild(barRow);
    });

    resultsContainer.appendChild(list);

    // Trigger bar animations after paint
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            document.querySelectorAll('.bar-fill').forEach(bar => {
                bar.style.width = bar.dataset.width + '%';
            });
        });
    });
}