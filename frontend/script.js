// ── API Configuration ────────────────────────────────────────────────────────
const API_BASE = "https://harshthorat85-alzheimers-mri-classifier.hf.space";

// ── Model Options ─────────────────────────────────────────────────────────────
// `id` must match a key in MODEL_CONFIGS in backend/app.py.
// Metrics are 5-fold subject-level cross-validation results from the study.
const MODELS = [
    {
        id: "corrected",
        name: "Corrected 3-class",
        tag: "Recommended",
        tagClass: "tag-good",
        description: "CN vs MCI vs AD, trained on 235 age-matched, CDR-rated subjects.",
        metrics: [["Accuracy", "51.9%"], ["Macro AUC", "0.718"]],
        disclaimer: "Trained on 235 age-matched OASIS-1 subjects (5-fold subject-level CV).",
    },
    {
        id: "binary",
        name: "Corrected binary",
        tag: "Age-matched",
        tagClass: "tag-good",
        description: "Cognitively normal vs impaired (CDR ≥ 0.5), same 235 subjects. Merging the small MCI and AD classes improves performance.",
        metrics: [["Macro F1", "0.684"], ["Macro AUC", "0.750"]],
        disclaimer: "Trained on 235 age-matched OASIS-1 subjects (135 CN, 100 impaired; 5-fold subject-level CV).",
    },
    {
        id: "original",
        name: "Original 3-class",
        tag: "Age-confounded",
        tagClass: "tag-warn",
        description: "Included 201 young adults with no CDR rating as cognitively normal, so it partly learned age instead of disease. Shown for comparison.",
        metrics: [["Accuracy", "79%"], ["Macro AUC", "0.811"]],
        disclaimer: "Trained on OASIS-1 data that included 201 unrated young adults as controls. Its accuracy is inflated by age confounding.",
    },
];

let selectedModel = MODELS[0].id;
let currentFile   = null;
let requestId     = 0;   // ignores stale responses if the model is switched mid-request

// ── DOM Elements ─────────────────────────────────────────────────────────────
const dropZone         = document.getElementById('dropZone');
const fileInput        = document.getElementById('fileInput');
const previewImage     = document.getElementById('previewImage');
const previewWrapper   = document.getElementById('previewWrapper');
const previewFilename  = document.getElementById('previewFilename');
const resultsContainer = document.getElementById('resultsContainer');
const warningBanner    = document.getElementById('warningBanner');
const modelPicker      = document.getElementById('modelPicker');
const resultModel      = document.getElementById('resultModel');
const disclaimerText   = document.getElementById('disclaimerText');
const changeLink       = document.getElementById('changeLink');

// ── Model Picker ──────────────────────────────────────────────────────────────
function renderPicker() {
    modelPicker.innerHTML = '';
    MODELS.forEach(m => {
        const option = document.createElement('label');
        option.className = 'model-option';
        option.dataset.id = m.id;

        const input = document.createElement('input');
        input.type = 'radio';
        input.name = 'model';
        input.value = m.id;
        input.checked = m.id === selectedModel;
        input.addEventListener('change', () => selectModel(m.id));

        const body = document.createElement('div');
        body.className = 'model-body';

        const head = document.createElement('div');
        head.className = 'model-head';
        const name = document.createElement('span');
        name.className = 'model-name';
        name.textContent = m.name;
        const tag = document.createElement('span');
        tag.className = `model-tag ${m.tagClass}`;
        tag.textContent = m.tag;
        head.append(name, tag);

        const desc = document.createElement('p');
        desc.className = 'model-desc';
        desc.textContent = m.description;

        const metrics = document.createElement('div');
        metrics.className = 'model-metrics';
        m.metrics.forEach(([label, value]) => {
            const span = document.createElement('span');
            span.innerHTML = `<b></b> `;
            span.querySelector('b').textContent = value;
            span.append(label);
            metrics.appendChild(span);
        });

        const unavailable = document.createElement('p');
        unavailable.className = 'model-unavailable';
        unavailable.textContent = 'Not available on the server right now.';

        body.append(head, desc, metrics, unavailable);
        option.append(input, body);
        modelPicker.appendChild(option);
    });
    updateDisclaimer();
}

function selectModel(id) {
    selectedModel = id;
    updateDisclaimer();
    if (currentFile) runPrediction(currentFile);   // re-run the same scan
}

function updateDisclaimer() {
    const m = MODELS.find(m => m.id === selectedModel);
    disclaimerText.textContent = m.disclaimer;
}

// Mark models the server could not load (non-blocking; the Space may be waking up)
async function checkAvailability() {
    try {
        const res = await fetch(`${API_BASE}/models`);
        if (!res.ok) return;
        const list = await res.json();
        list.forEach(({ id, available }) => {
            const option = modelPicker.querySelector(`[data-id="${id}"]`);
            if (!option) return;
            option.classList.toggle('disabled', !available);
            option.querySelector('input').disabled = !available;
        });
    } catch (e) {
        console.warn('Could not check model availability:', e);
    }
}

// ── Drag and Drop ─────────────────────────────────────────────────────────────
['dragenter', 'dragover'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });
});

['dragleave', 'drop'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
    });
});

dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) handleImageUpload(files[0]);
});

const openPicker = () => fileInput.click();
dropZone.addEventListener('click', openPicker);
changeLink.addEventListener('click', openPicker);
[dropZone, changeLink].forEach(el => el.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openPicker(); }
}));

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) handleImageUpload(fileInput.files[0]);
    fileInput.value = '';   // allows re-selecting the same file
});

// ── Upload Handler ────────────────────────────────────────────────────────────
function handleImageUpload(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload a valid image file (JPEG, PNG, or TIFF).');
        return;
    }
    currentFile = file;

    const reader = new FileReader();
    reader.onloadend = () => {
        previewImage.src = reader.result;
        previewWrapper.style.display = 'flex';
        dropZone.style.display       = 'none';
        previewFilename.textContent  = file.name;
    };
    reader.readAsDataURL(file);

    runPrediction(file);
}

// ── Prediction ────────────────────────────────────────────────────────────────
async function runPrediction(file) {
    const thisRequest = ++requestId;
    const model = selectedModel;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', model);

    warningBanner.style.display = 'none';
    resultModel.style.display   = 'none';
    resultsContainer.innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
            Running ${MODELS.find(m => m.id === model).name} model…
        </div>`;

    try {
        const response = await fetch(`${API_BASE}/predict`, { method: 'POST', body: formData });
        if (thisRequest !== requestId) return;

        if (!response.ok) {
            let detail = response.statusText;
            try { detail = (await response.json()).detail || detail; } catch (_) {}
            throw new Error(`Server error ${response.status}: ${detail}`);
        }

        const data = await response.json();
        if (thisRequest !== requestId) return;
        renderResults(data);

    } catch (error) {
        if (thisRequest !== requestId) return;
        console.error('API Error:', error);
        resultsContainer.innerHTML = `
            <div class="error">
                ❌ Could not get a prediction. If the server was asleep, wait a minute and try again.
                <small></small>
            </div>`;
        resultsContainer.querySelector('small').textContent = error.message;
    }
}

// ── Render Results ────────────────────────────────────────────────────────────
function renderResults(data) {
    resultsContainer.innerHTML = '';

    resultModel.textContent   = `Result from: ${data.model_name}`;
    resultModel.style.display = 'block';

    if (data.low_confidence_warning) {
        warningBanner.innerText = "⚠️ Low confidence: no class scored clearly above the others for this model. Treat this result with caution.";
        warningBanner.style.display = 'block';
    }

    const list = document.createElement('div');
    list.className = 'results-list';

    Object.entries(data.confidence_scores).forEach(([className, score]) => {
        const percentage = (score * 100).toFixed(1);
        const isTop      = className === data.prediction;

        const barRow = document.createElement('div');
        barRow.className = `bar-row ${isTop ? 'winner' : ''}`;
        barRow.innerHTML = `
            <div class="bar-labels">
                <span class="class-name"></span>
                <span class="percentage-val">${percentage}%</span>
            </div>
            <div class="bar-track">
                <div class="bar-fill" data-width="${percentage}"></div>
            </div>
        `;
        const nameEl = barRow.querySelector('.class-name');
        nameEl.textContent = className;
        if (isTop) {
            const badge = document.createElement('span');
            badge.className = 'prediction-badge';
            badge.textContent = 'top prediction';
            nameEl.appendChild(badge);
        }
        list.appendChild(barRow);
    });

    resultsContainer.appendChild(list);

    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            document.querySelectorAll('.bar-fill').forEach(bar => {
                bar.style.width = bar.dataset.width + '%';
            });
        });
    });
}

// ── Init ──────────────────────────────────────────────────────────────────────
renderPicker();
checkAvailability();
