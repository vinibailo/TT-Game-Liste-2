let cropper = null;
let currentIndex = 0;
let currentId = null;
let currentIgdbId = null;
let currentUpload = null;
let originalImage = null;
let navigating = false;
let totalGames = 0;
let toastTimeout = null;
const lookupConfigs = {
    Developers: { endpoint: 'developers', elementId: 'developers' },
    Publishers: { endpoint: 'publishers', elementId: 'publishers' },
    Genres: { endpoint: 'genres', elementId: 'genres' },
    GameModes: { endpoint: 'game_modes', elementId: 'modes' },
    Platforms: { endpoint: 'platforms', elementId: 'platforms' },
};
const lookupChoicesInstances = {};
const lookupIdToName = {};
const lookupNameToId = {};
const imageUploadInput = document.getElementById('imageUpload');
const placeholderImage = '/no-image.jpg';
const saveButton = document.getElementById('save');
const saveButtonLabel = saveButton ? saveButton.querySelector('.btn-label') : null;
const saveButtonDefaultLabel = saveButtonLabel
    ? saveButtonLabel.textContent.trim()
    : (saveButton ? saveButton.textContent.trim() : '');
const imageResolutionLabel = document.getElementById('image-resolution');
const resolutionWarningTag = document.getElementById('image-warning');
const LOW_RESOLUTION_WARNING = 'Imagem menor que 1080px será ampliada.';
const jumpForm = document.getElementById('jump-form');
const jumpInput = document.getElementById('jump-input');
const jumpSubmit = document.getElementById('jump-submit');
const urlSearchParams = new URLSearchParams(window.location.search);
const initialGameIdParam = (() => {
    const value = urlSearchParams.get('game_id');
    if (!value) {
        return null;
    }
    const trimmed = value.trim();
    if (!/^\d+$/.test(trimmed)) {
        return null;
    }
    const parsed = Number.parseInt(trimmed, 10);
    return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
})();

function setSaveButtonLabel(text) {
    if (!saveButton) return;
    if (saveButtonLabel) {
        saveButtonLabel.textContent = text;
    } else {
        saveButton.textContent = text;
    }
}

function showResolutionWarning(message = LOW_RESOLUTION_WARNING) {
    if (!resolutionWarningTag) return;
    resolutionWarningTag.textContent = message;
    resolutionWarningTag.classList.add('visible');
}

function hideResolutionWarning() {
    if (!resolutionWarningTag) return;
    resolutionWarningTag.textContent = '';
    resolutionWarningTag.classList.remove('visible');
}

hideResolutionWarning();
const categoriesList = window.categoriesList || [];

function populateSelect(id, options) {
    const sel = document.getElementById(id);
    if (!sel) return;
    sel.innerHTML = '';
    options.forEach(o => {
        const opt = document.createElement('option');
        opt.value = o;
        opt.textContent = o;
        sel.appendChild(opt);
    });
}

function normalizeLookupName(value) {
    if (value == null) return '';
    return String(value).trim();
}

function lookupNameFingerprint(value) {
    const normalized = normalizeLookupName(value);
    return normalized ? normalized.toLocaleLowerCase() : '';
}

async function fetchLookupOptions(type) {
    const response = await fetch(`/api/lookups/${type}`);
    if (!response.ok) {
        throw new Error(`Failed to load lookup options for ${type}`);
    }
    const payload = await response.json();
    const items = Array.isArray(payload.items) ? payload.items : [];
    return items
        .map(item => {
            if (!item || item.id == null) {
                return null;
            }
            const name = normalizeLookupName(item.name);
            const idNumber = Number.parseInt(item.id, 10);
            if (!name || Number.isNaN(idNumber)) {
                return null;
            }
            return { id: idNumber, name };
        })
        .filter(Boolean);
}

function populateLookupSelectElement(element, items) {
    element.innerHTML = '';
    const idMap = new Map();
    const nameIndex = new Map();
    items.forEach(item => {
        const value = String(item.id);
        const option = document.createElement('option');
        option.value = value;
        option.textContent = item.name;
        element.appendChild(option);
        idMap.set(value, item.name);
        const fingerprint = lookupNameFingerprint(item.name);
        if (fingerprint && !nameIndex.has(fingerprint)) {
            nameIndex.set(fingerprint, value);
        }
    });
    return { idMap, nameIndex };
}

function setLookupSelection(key, values) {
    const instance = lookupChoicesInstances[key];
    if (!instance) return;
    const normalized = Array.isArray(values)
        ? values
              .map(value => String(value).trim())
              .filter(Boolean)
        : [];
    instance.removeActiveItems();
    if (!normalized.length) {
        return;
    }
    if (typeof instance.setChoiceByValue === 'function') {
        instance.setChoiceByValue(normalized);
    } else {
        instance.setValue(normalized);
    }
}

function mapNamesToLookupIds(key, names) {
    const index = lookupNameToId[key];
    if (!index) return [];
    const seen = new Set();
    const results = [];
    names.forEach(name => {
        const fingerprint = lookupNameFingerprint(name);
        if (!fingerprint) {
            return;
        }
        const value = index.get(fingerprint);
        if (value && !seen.has(value)) {
            seen.add(value);
            results.push(value);
        }
    });
    return results;
}

function coerceNameArray(value) {
    if (!value) return [];
    if (Array.isArray(value)) {
        return value.map(normalizeLookupName).filter(Boolean);
    }
    if (typeof value === 'string') {
        return value
            .split(',')
            .map(normalizeLookupName)
            .filter(Boolean);
    }
    return [];
}

function extractLookupIds(key, selection, fallbackValue) {
    const idMap = lookupIdToName[key] || new Map();
    const ids = [];
    const seen = new Set();
    if (selection && Array.isArray(selection.ids)) {
        selection.ids.forEach(idValue => {
            if (idValue == null) return;
            const strValue = String(idValue).trim();
            if (!strValue || seen.has(strValue)) {
                return;
            }
            if (idMap.size && !idMap.has(strValue)) {
                return;
            }
            seen.add(strValue);
            ids.push(strValue);
        });
    }
    if (!ids.length && selection && Array.isArray(selection.selected)) {
        selection.selected.forEach(item => {
            if (!item) return;
            if (item.id == null) return;
            const strValue = String(item.id).trim();
            if (!strValue || seen.has(strValue)) return;
            if (idMap.size && !idMap.has(strValue)) {
                return;
            }
            seen.add(strValue);
            ids.push(strValue);
        });
    }
    if (!ids.length) {
        let fallbackNames = [];
        if (selection && Array.isArray(selection.names) && selection.names.length) {
            fallbackNames = selection.names;
        } else if (fallbackValue !== undefined) {
            fallbackNames = coerceNameArray(fallbackValue);
        }
        if (fallbackNames.length) {
            const mapped = mapNamesToLookupIds(key, fallbackNames);
            mapped.forEach(value => {
                if (!seen.has(value)) {
                    seen.add(value);
                    ids.push(value);
                }
            });
        }
    }
    return ids;
}

function applyLookupSelection(key, selection, fallbackValue) {
    const values = extractLookupIds(key, selection, fallbackValue);
    setLookupSelection(key, values);
}

function buildLookupSelections() {
    const result = {};
    Object.keys(lookupConfigs).forEach(key => {
        const instance = lookupChoicesInstances[key];
        const idMap = lookupIdToName[key] || new Map();
        const selection = {
            ids: [],
            names: [],
            selected: [],
            display: '',
        };
        if (instance && typeof instance.getValue === 'function') {
            const rawValues = instance.getValue(true);
            const seenIds = new Set();
            if (Array.isArray(rawValues)) {
                rawValues.forEach(value => {
                    if (value == null) return;
                    const strValue = String(value).trim();
                    if (!strValue) return;
                    const label = idMap.get(strValue) || '';
                    const parsed = Number.parseInt(strValue, 10);
                    const numericId = Number.isNaN(parsed) ? null : parsed;
                    if (numericId !== null) {
                        if (seenIds.has(numericId)) {
                            return;
                        }
                        seenIds.add(numericId);
                        selection.ids.push(numericId);
                    }
                    if (label) {
                        selection.names.push(label);
                    }
                    selection.selected.push({
                        id: numericId,
                        name: label || null,
                    });
                });
            }
        }
        selection.display = selection.names.join(', ');
        result[key] = selection;
    });
    return result;
}

async function initializeLookupControls() {
    const entries = await Promise.all(
        Object.entries(lookupConfigs).map(async ([key, config]) => {
            try {
                const items = await fetchLookupOptions(config.endpoint);
                return { key, config, items };
            } catch (err) {
                console.error(`Failed to load lookup options for ${config.endpoint}`, err);
                return { key, config, items: [] };
            }
        })
    );
    entries.forEach(({ key, config, items }) => {
        const element = document.getElementById(config.elementId);
        if (!element) {
            lookupChoicesInstances[key] = null;
            lookupIdToName[key] = new Map();
            lookupNameToId[key] = new Map();
            return;
        }
        const { idMap, nameIndex } = populateLookupSelectElement(element, items);
        lookupIdToName[key] = idMap;
        lookupNameToId[key] = nameIndex;
        const choices = new Choices(element, {
            removeItemButton: true,
            shouldSort: false,
        });
        choices.passedElement.element.addEventListener('change', saveSession);
        lookupChoicesInstances[key] = choices;
    });
}

function updateGameIdDisplay(idValue) {
    const idText = idValue ? String(idValue) : '—';
    document.getElementById('game-id').textContent = idText;
}

function updateIgdbIdDisplay(idValue) {
    const span = document.getElementById('igdb-id');
    if (!span) return;
    const idText = idValue ? String(idValue) : '—';
    span.textContent = idText;
}

function collectFields() {
    const lookups = buildLookupSelections();
    return {
        Name: document.getElementById('name').value,
        Summary: document.getElementById('summary').value,
        FirstLaunchDate: document.getElementById('first-launch').value,
        Category: document.getElementById('category').value,
        Lookups: lookups,
    };
}

function setNavDisabled(state) {
    ['next', 'previous', 'skip', 'reset'].forEach(id => {
        const button = document.getElementById(id);
        if (button) {
            button.disabled = state;
        }
    });
}

function setJumpControlsDisabled(state) {
    if (jumpInput) {
        jumpInput.disabled = state;
    }
    if (jumpSubmit) {
        jumpSubmit.disabled = state;
    }
}

function isTypingElement(element) {
    if (!element) return false;
    if (element.isContentEditable) return true;
    const tag = element.tagName;
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
}

function saveSession() {
    const imgSrc = document.getElementById('image').src;
    const data = {
        index: currentIndex,
        id: currentId,
        igdbId: currentIgdbId,
        fields: collectFields(),
        image: imgSrc && imgSrc !== placeholderImage ? imgSrc : '',
        upload_name: currentUpload
    };
    localStorage.setItem('session', JSON.stringify(data));
}

function restoreSession() {
    const s = localStorage.getItem('session');
    if (!s) return;
    const data = JSON.parse(s);
    if (data.index !== currentIndex) return;
    currentId = data.id != null ? String(data.id) : null;
    updateGameIdDisplay(currentId);
    if (Object.prototype.hasOwnProperty.call(data, 'igdbId')) {
        const storedIgdb = data.igdbId;
        currentIgdbId = storedIgdb != null && storedIgdb !== '' ? String(storedIgdb) : null;
        updateIgdbIdDisplay(currentIgdbId);
    }
    document.getElementById('name').value = data.fields.Name;
    document.getElementById('game-name').textContent = data.fields.Name || 'Untitled Game';
    const summaryEl = document.getElementById('summary');
    summaryEl.value = data.fields.Summary;
    document.getElementById('first-launch').value = data.fields.FirstLaunchDate;
    document.getElementById('category').value = data.fields.Category;
    const lookups = (data.fields && data.fields.Lookups) || {};
    applyLookupSelection('Developers', lookups.Developers, data.fields.Developers);
    applyLookupSelection('Publishers', lookups.Publishers, data.fields.Publishers);
    applyLookupSelection('Genres', lookups.Genres, data.fields.Genres);
    applyLookupSelection('GameModes', lookups.GameModes, data.fields.GameModes);
    applyLookupSelection('Platforms', lookups.Platforms, data.fields.Platforms);
    if (data.image) setImage(data.image);
    currentUpload = data.upload_name;
}

function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    if (!toast) return;
    const normalizedType = type === 'error' ? 'warning' : type;
    toast.textContent = message;
    toast.className = '';
    void toast.offsetWidth;
    toast.classList.add(normalizedType, 'show');
    clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => {
        toast.classList.remove('show');
    }, 3200);
}

function generateSummary() {
    const name = document.getElementById('name').value;
    const btn = document.getElementById('generate-summary');
    btn.disabled = true;
    btn.textContent = 'Gerando...';
    fetch('api/summary', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({game_name: name})
    })
    .then(r => r.json())
    .then(res => {
        if (res.summary) {
            document.getElementById('summary').value = res.summary;
            saveSession();
        } else if (res.error) {
            console.error(res.error);
            showToast(res.error, 'warning');
        } else {
            showToast('Não foi possível gerar o resumo.', 'warning');
        }
    })
    .catch(err => { console.error(err); showToast('Erro ao gerar resumo.', 'warning'); })
    .finally(() => {
        btn.disabled = false;
        btn.textContent = 'Gerar Resumo';
    });
}

function updatePreview() {
    if (!cropper) return;
    const canvas = cropper.getCroppedCanvas({width:1080, height:1080});
    if (!canvas) {
        showToast('Não foi possível gerar a pré-visualização da imagem.', 'warning');
        return;
    }
    document.getElementById('preview').src = canvas.toDataURL('image/jpeg', 0.9);
    saveSession();
}

function setImage(dataUrl) {
    const img = document.getElementById('image');
    const saveBtn = saveButton;
    // Disable saving while the image is being prepared
    if (saveBtn) {
        saveBtn.disabled = true;
    }
    setSaveButtonLabel('Loading...');
    hideResolutionWarning();

    img.onload = function(){
        try {
            if (cropper) cropper.destroy();
            cropper = new Cropper(img, {
                aspectRatio: 1,
                viewMode: 2,
                autoCropArea: 1,
                modal: false,
                crop: updatePreview,
                ready: updatePreview,
                background: false
            });
            const isLowResolution = Math.min(img.naturalWidth, img.naturalHeight) < 1080;
            if (isLowResolution) {
                showResolutionWarning();
            } else {
                hideResolutionWarning();
            }
            if (imageResolutionLabel) {
                imageResolutionLabel.textContent = `${img.naturalWidth}x${img.naturalHeight}`;
            }
        } catch (err) {
            console.error(err);
            showToast('Failed to initialize image: ' + (err.message || err), 'warning');
            hideResolutionWarning();
        } finally {
            // Re-enable the save button after cropper is ready or on error
            if (saveBtn) {
                saveBtn.disabled = false;
            }
            setSaveButtonLabel(saveButtonDefaultLabel);
        }
    };
    img.onerror = function(){
        showToast('Não foi possível carregar a imagem enviada.', 'warning');
        if (saveBtn) {
            saveBtn.disabled = false;
        }
        setSaveButtonLabel(saveButtonDefaultLabel);
        hideResolutionWarning();
        if (imageResolutionLabel) {
            imageResolutionLabel.textContent = '';
        }
    };
    img.src = dataUrl;
    imageUploadInput.value = '';
    if (img.complete) {
        img.naturalWidth ? img.onload() : img.onerror();
    }
}

function clearImage() {
    if (cropper) { cropper.destroy(); cropper = null; }
    document.getElementById('image').src = '';
    document.getElementById('preview').src = '';
    if (imageResolutionLabel) {
        imageResolutionLabel.textContent = '';
    }
    currentUpload = null;
    imageUploadInput.value = '';
    hideResolutionWarning();
    saveSession();
}

function applyGameData(data) {
    if (data.done) {
        document.body.innerHTML = `<h2>${data.message}</h2>`;
        return;
    }
    currentIndex = data.index;
    currentId = data.id != null ? String(data.id) : null;
    const igdbValue = data.game && data.game.IGDBID != null
        ? String(data.game.IGDBID).trim()
        : '';
    currentIgdbId = igdbValue ? igdbValue : null;
    document.getElementById('game-name').textContent = data.game.Name || 'Untitled Game';
    updateGameIdDisplay(currentId);
    updateIgdbIdDisplay(currentIgdbId);
    const processed = Math.max(0, (data.seq || 1) - 1);
    const total = data.total || 0;
    totalGames = total;
    const safeProcessed = total ? Math.min(processed, total) : processed;
    const ratio = total ? safeProcessed / total : 0;
    const progressPercent = Math.min(100, Math.max(0, ratio * 100));
    document.getElementById('progress').style.width = `${progressPercent}%`;
    const countLabel = `${safeProcessed}/${total || 0}`;
    const percentLabel = `${progressPercent.toFixed(2)}%`;
    document.getElementById('progress-count').textContent = countLabel;
    document.getElementById('progress-percent').textContent = percentLabel;
    document.getElementById('name').value = data.game.Name || '';
    const summaryEl = document.getElementById('summary');
    summaryEl.value = data.game.Summary || '';
    document.getElementById('first-launch').value = data.game.FirstLaunchDate || '';
    document.getElementById('category').value = data.game.Category || '';
    const lookupSelections = (data.game && data.game.Lookups) || {};
    applyLookupSelection('Developers', lookupSelections.Developers, data.game.Developers);
    applyLookupSelection('Publishers', lookupSelections.Publishers, data.game.Publishers);
    applyLookupSelection('Genres', lookupSelections.Genres, data.game.Genres);
    applyLookupSelection('GameModes', lookupSelections.GameModes, data.game.GameModes);
    applyLookupSelection('Platforms', lookupSelections.Platforms, data.game.Platforms);
    if (data.cover) {
        setImage(data.cover);
        originalImage = data.cover;
    } else {
        clearImage();
        document.getElementById('image').src = placeholderImage;
        document.getElementById('preview').src = placeholderImage;
        originalImage = null;
    }
    currentUpload = null;
    restoreSession();
    if (Array.isArray(data.missing) && data.missing.length) {
        showToast('Campos vazios: ' + data.missing.join(', '), 'warning');
    }
    saveSession();
}

function loadGame() {
    setNavDisabled(true);
    return fetch('api/game')
        .then(r => r.json())
        .then(applyGameData)
        .catch(err => {
            console.error(err.stack || err);
            showToast('Failed to load game: ' + err.message, 'warning');
        })
        .finally(() => setNavDisabled(false));
}

async function loadInitialGame() {
    if (initialGameIdParam) {
        navigating = true;
        setNavDisabled(true);
        setJumpControlsDisabled(true);
        try {
            await requestGameById(initialGameIdParam);
        } catch (err) {
            console.error(err);
            const code = err && (err.code || err.message);
            const message = translateJumpError(code);
            showToast(message, 'warning');
            await loadGame();
        } finally {
            navigating = false;
            setNavDisabled(false);
            setJumpControlsDisabled(false);
        }
        return;
    }
    await loadGame();
}

async function saveGame() {
    if (!cropper) {
        showToast('Selecione uma imagem', 'warning');
        return;
    }
    const canvas = cropper.getCroppedCanvas({width:1080, height:1080});
    if (!canvas) {
        showToast('Não foi possível preparar a imagem.', 'warning');
        return;
    }
    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
    const fields = collectFields();
    setNavDisabled(true);
    const saveBtn = saveButton;
    if (saveBtn) {
        saveBtn.disabled = true;
    }
    setSaveButtonLabel('Saving...');
    const pendingUpload = currentUpload;
    const attemptSave = async (allowRetry = true) => {
        const payload = {
            index: currentIndex,
            id: currentId,
            fields,
            image: dataUrl,
            upload_name: pendingUpload
        };
        const response = await fetch('api/save', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        const result = await response.json();
        if (response.status === 409 && result && result.error === 'index mismatch') {
            if (Object.prototype.hasOwnProperty.call(result, 'expected')) {
                const nextIndex = Number(result.expected);
                if (!Number.isNaN(nextIndex)) {
                    currentIndex = nextIndex;
                }
            }
            let expectedIdStr = null;
            if (Object.prototype.hasOwnProperty.call(result, 'expected_id')) {
                const expectedIdValue = result.expected_id;
                if (expectedIdValue !== null && expectedIdValue !== undefined) {
                    expectedIdStr = String(expectedIdValue);
                }
            }
            currentId = expectedIdStr;
            updateGameIdDisplay(currentId);
            currentIgdbId = null;
            updateIgdbIdDisplay(currentIgdbId);
            await loadGame();
            setNavDisabled(true);
            const refreshedId = currentId != null ? String(currentId) : null;
            if (!allowRetry) {
                showToast('Save aborted: the record changed again. Review the refreshed data before saving once more.', 'warning');
                const err = new Error('index mismatch');
                err.handled = true;
                throw err;
            }
            if (!expectedIdStr) {
                showToast('Save aborted: unable to determine the expected record. Review the refreshed data before trying again.', 'warning');
                const err = new Error('missing expected id');
                err.handled = true;
                throw err;
            }
            if (refreshedId !== expectedIdStr) {
                showToast('Save aborted: the record changed while saving. Review the refreshed data before trying again.', 'warning');
                const err = new Error('stale data mismatch');
                err.handled = true;
                throw err;
            }
            return attemptSave(false);
        }
        if (!response.ok || result.error) {
            throw new Error(result.error || 'save failed');
        }
        return result;
    };
    try {
        await attemptSave(true);
        localStorage.removeItem('session');
        currentUpload = null;
        showToast(`Saved ✔ Game ${currentIndex + 1} of ${totalGames}`, 'success');
        await nextGame(true);
    } catch (err) {
        console.error(err);
        if (!err || !err.handled) {
            showToast('Failed to save game: ' + err.message, 'warning');
        }
    } finally {
        if (saveBtn) {
            saveBtn.disabled = false;
        }
        setSaveButtonLabel(saveButtonDefaultLabel);
        setNavDisabled(false);
    }
}

async function skipGame() {
    setNavDisabled(true);
    try {
        const response = await fetch('api/skip', {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({index: currentIndex, upload_name: currentUpload})
        });
        const result = await response.json();
        if (!response.ok || result.error) {
            throw new Error(result.error || 'skip failed');
        }
        localStorage.removeItem('session');
        currentUpload = null;
        await loadGame();
    } catch (err) {
        console.error(err);
        showToast('Failed to skip game: ' + err.message, 'warning');
    } finally {
        setNavDisabled(false);
    }
}

function translateJumpError(code) {
    switch (code) {
        case 'missing id':
        case 'invalid id':
            return 'Digite um ID válido.';
        case 'id not found':
            return 'ID não encontrado.';
        case 'invalid index':
            return 'Registro não disponível.';
        case 'invalid source index':
            return 'Não foi possível localizar o índice desse ID.';
        default:
            return 'Não foi possível carregar o jogo solicitado.';
    }
}

async function goToGameById(event) {
    if (event) {
        event.preventDefault();
    }
    if (!jumpInput) {
        return;
    }
    const rawValue = jumpInput.value.trim();
    if (!rawValue) {
        showToast('Digite um ID válido.', 'warning');
        jumpInput.focus();
        return;
    }
    if (!/^\d+$/.test(rawValue)) {
        showToast('Digite um ID válido.', 'warning');
        jumpInput.focus();
        jumpInput.select();
        return;
    }
    const idValue = Number.parseInt(rawValue, 10);
    if (!Number.isInteger(idValue) || idValue < 1) {
        showToast('Digite um ID válido.', 'warning');
        jumpInput.focus();
        jumpInput.select();
        return;
    }
    if (navigating) {
        return;
    }
    navigating = true;
    setNavDisabled(true);
    setJumpControlsDisabled(true);
    let shouldRefocus = false;
    try {
        await requestGameById(idValue);
        jumpInput.value = '';
    } catch (err) {
        console.error(err);
        const code = err && (err.code || err.message);
        const message = translateJumpError(code);
        showToast(message, 'warning');
        shouldRefocus = true;
    } finally {
        navigating = false;
        setJumpControlsDisabled(false);
        setNavDisabled(false);
        if (shouldRefocus) {
            jumpInput.focus();
            jumpInput.select();
        }
    }
}

async function requestGameById(idValue) {
    const response = await fetch('api/game_by_id', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({id: idValue, upload_name: currentUpload})
    });
    let payload = null;
    try {
        payload = await response.json();
    } catch (err) {
        payload = null;
    }
    if (!response.ok || !payload || payload.error) {
        const errorCode = payload && payload.error ? payload.error : 'jump failed';
        const error = new Error(errorCode);
        error.code = errorCode;
        throw error;
    }
    localStorage.removeItem('session');
    currentUpload = null;
    applyGameData(payload);
}

function nextGame(autoAdvance = false) {
    if (navigating) return Promise.resolve();
    navigating = true;
    if (!autoAdvance) {
        setNavDisabled(true);
    }
    return fetch('api/next', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({upload_name: currentUpload})
    })
      .then(r=>r.json())
      .then(data => {
          localStorage.removeItem('session');
          applyGameData(data);
      })
      .catch(err => {
          console.error(err);
          showToast('Failed to move to next game: ' + err.message, 'warning');
      })
      .finally(() => {
          navigating = false;
          if (!autoAdvance) {
              setNavDisabled(false);
          }
      });
}

function previousGame() {
    if (navigating) return;
    navigating = true;
    setNavDisabled(true);
    fetch('api/back', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({upload_name: currentUpload})
    })
      .then(r=>r.json()).then(data => { localStorage.removeItem('session'); applyGameData(data); })
      .catch(err => {
          console.error(err);
          showToast('Failed to move to previous game: ' + err.message, 'warning');
      })
      .finally(() => {
          navigating = false;
          setNavDisabled(false);
      });
}

function resetFields() {
    fetch(`api/game/${currentIndex}/raw`).then(r=>r.json()).then(data=>{
        document.getElementById('name').value = data.game.Name || '';
        document.getElementById('game-name').textContent = data.game.Name || 'Untitled Game';
        const summaryEl = document.getElementById('summary');
        summaryEl.value = data.game.Summary || '';
        document.getElementById('first-launch').value = data.game.FirstLaunchDate || '';
        document.getElementById('category').value = data.game.Category || '';
        applyLookupSelection('Developers', undefined, data.game.Developers);
        applyLookupSelection('Publishers', undefined, data.game.Publishers);
        applyLookupSelection('Genres', undefined, data.game.Genres);
        applyLookupSelection('GameModes', undefined, data.game.GameModes);
        applyLookupSelection('Platforms', undefined, data.game.Platforms);
        if (data.cover) {
            setImage(data.cover);
            originalImage = data.cover;
        } else {
            clearImage();
            document.getElementById('image').src = placeholderImage;
            document.getElementById('preview').src = placeholderImage;
            originalImage = null;
        }
        currentUpload = null;
        saveSession();
    }).catch(err => {
        console.error(err);
        showToast('Failed to reset fields: ' + err.message, 'warning');
    });
}

imageUploadInput.addEventListener('change', function(){
    const file = this.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    fetch('api/upload', {method:'POST', body: formData})
        .then(r=>r.json())
        .then(res => { currentUpload = res.filename; setImage(res.data); })
        .catch(err => {
            console.error(err);
            showToast('Failed to upload image: ' + err.message, 'warning');
        });
});

document.getElementById('generate-summary').addEventListener('click', generateSummary);
document.getElementById('save').addEventListener('click', saveGame);
document.getElementById('skip').addEventListener('click', skipGame);
document.getElementById('next').addEventListener('click', () => nextGame());
document.getElementById('previous').addEventListener('click', previousGame);
document.getElementById('reset').addEventListener('click', resetFields);
if (jumpForm) {
    jumpForm.addEventListener('submit', goToGameById);
}
function revertImage() {
    fetch(`api/game/${currentIndex}/raw`).then(r=>r.json()).then(data=>{
        if (data.cover) {
            setImage(data.cover);
            originalImage = data.cover;
        } else {
            clearImage();
            document.getElementById('image').src = placeholderImage;
            document.getElementById('preview').src = placeholderImage;
            originalImage = null;
        }
        currentUpload = null;
        saveSession();
    }).catch(err => {
        console.error(err);
        showToast('Failed to revert image: ' + err.message, 'warning');
    });
}

document.getElementById('revert-image').addEventListener('click', revertImage);
['name','summary','first-launch','developers','publishers','category'].forEach(id => {
    const el = document.getElementById(id);
    ['change','input'].forEach(ev => el.addEventListener(ev, saveSession));
});

document.getElementById('name').addEventListener('input', (event) => {
    const value = event.target.value;
    document.getElementById('game-name').textContent = value || 'Untitled Game';
});

document.addEventListener('keydown', (event) => {
    const activeElement = document.activeElement;
    const typing = isTypingElement(activeElement);
    const key = event.key;
    const lower = key.toLowerCase();

    if (key === 'Escape') {
        event.preventDefault();
        resetFields();
        return;
    }

    if (lower === 's') {
        if (event.ctrlKey || event.metaKey) {
            event.preventDefault();
            saveGame();
            return;
        }
        if (!event.altKey && !event.metaKey && !event.ctrlKey && !typing) {
            event.preventDefault();
            saveGame();
            return;
        }
    }

    if (!event.altKey && !event.ctrlKey && !event.metaKey && !typing) {
        if (key === 'ArrowRight') {
            event.preventDefault();
            nextGame();
        } else if (key === 'ArrowLeft') {
            event.preventDefault();
            previousGame();
        }
    }
});

populateSelect('category', categoriesList);

initializeLookupControls()
    .catch(err => {
        console.error(err);
        showToast('Failed to load lookup options.', 'warning');
    })
    .finally(() => {
        loadInitialGame();
    });
