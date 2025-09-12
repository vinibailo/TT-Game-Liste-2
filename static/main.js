let cropper = null;
let currentIndex = 0;
let currentId = null;
let currentUpload = null;
let originalImage = null;
let genresChoices, modesChoices;
let navigating = false;
const imageUploadInput = document.getElementById('imageUpload');
const placeholderImage = 'https://i.imgur.com/XZvvGuQ.png';
const genresList = [
    'Ação e Aventura',
    'Cartas e Tabuleiro',
    'Clássicos',
    'Família e Crianças',
    'Luta',
    'Indie',
    'Multijogador',
    'Plataformas',
    'Quebra-cabeça e Trivia',
    'Corrida e Voo',
    'RPG',
    'Tiro',
    'Simulação',
    'Esportes',
    'Estratégia',
    'Horror de Sobrevivência',
    'Mundo Aberto',
    'Outros'
];
const modesList = [
    'Single-player',
    'Multiplayer local',
    'Multiplayer online',
    'Cooperativo (Co-op)',
    'Competitivo (PvP)'
];

function populateSelect(id, options) {
    const sel = document.getElementById(id);
    options.forEach(o => {
        const opt = document.createElement('option');
        opt.value = o; opt.textContent = o;
        sel.appendChild(opt);
    });
}

function setChoices(instance, values) {
    instance.removeActiveItems();
    instance.setValue(values || []);
}

function collectFields() {
    return {
        Name: document.getElementById('name').value,
        Summary: document.getElementById('summary').value,
        FirstLaunchDate: document.getElementById('first-launch').value,
        Developers: document.getElementById('developers').value,
        Publishers: document.getElementById('publishers').value,
        Genres: genresChoices.getValue(true),
        GameModes: modesChoices.getValue(true)
    };
}

function setNavDisabled(state) {
    document.getElementById('next').disabled = state;
    document.getElementById('previous').disabled = state;
}

function saveSession() {
    const imgSrc = document.getElementById('image').src;
    const data = {
        index: currentIndex,
        id: currentId,
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
    currentId = data.id;
    document.getElementById('game-id').textContent = `ID: ${currentId || ''}`;
    document.getElementById('name').value = data.fields.Name;
    document.getElementById('summary').value = data.fields.Summary;
    document.getElementById('first-launch').value = data.fields.FirstLaunchDate;
    document.getElementById('developers').value = data.fields.Developers;
    document.getElementById('publishers').value = data.fields.Publishers;
    setChoices(genresChoices, data.fields.Genres);
    setChoices(modesChoices, data.fields.GameModes);
    if (data.image) setImage(data.image);
    currentUpload = data.upload_name;
}

function showAlert(message, type = 'success') {
    const banner = document.getElementById('alert-banner');
    banner.textContent = message;
    banner.className = type;
    banner.style.display = 'block';
    setTimeout(() => {
        banner.style.display = 'none';
    }, 5000);
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
            alert(res.error);
        } else {
            alert('Não foi possível gerar o resumo.');
        }
    })
    .catch(err => { console.error(err); alert('Erro ao gerar resumo.'); })
    .finally(() => {
        btn.disabled = false;
        btn.textContent = 'Gerar Resumo';
    });
}

function updatePreview() {
    if (!cropper) return;
    const canvas = cropper.getCroppedCanvas({width:1080, height:1080});
    document.getElementById('preview').src = canvas.toDataURL('image/jpeg', 0.9);
    saveSession();
}

function setImage(dataUrl) {
    const img = document.getElementById('image');
    const saveBtn = document.getElementById('save');
    // Disable saving while the image is being prepared
    saveBtn.disabled = true;
    const originalSaveText = saveBtn.textContent;
    saveBtn.textContent = 'Loading...';

    img.onload = function(){
        if (cropper) cropper.destroy();
        cropper = new Cropper(img, {
            aspectRatio: 1,
            viewMode: 2,
            autoCropArea: 1,
            modal: false,
            crop: updatePreview,
            background: false
        });
        if (Math.min(img.naturalWidth, img.naturalHeight) < 1080) {
            showAlert('Imagem menor que 1080px será ampliada.', 'warning');
        }
        document.getElementById('image-resolution').textContent = `${img.naturalWidth}x${img.naturalHeight}`;
        updatePreview();

        // Re-enable the save button after cropper is ready
        saveBtn.disabled = false;
        saveBtn.textContent = originalSaveText;
    };
    img.src = dataUrl;
    imageUploadInput.value = '';
}

function clearImage() {
    if (cropper) { cropper.destroy(); cropper = null; }
    document.getElementById('image').src = '';
    document.getElementById('preview').src = '';
    document.getElementById('image-resolution').textContent = '';
    currentUpload = null;
    imageUploadInput.value = '';
    saveSession();
}

function applyGameData(data) {
    if (data.done) {
        document.body.innerHTML = `<h2>${data.message}</h2>`;
        return;
    }
    currentIndex = data.index;
    currentId = data.id;
    document.getElementById('game-name').textContent = data.game.Name || '';
    const seq = data.seq || 0;
    const total = data.total || 1;
    document.getElementById('progress-text').textContent = `${seq} / ${total} | ${(seq/total*100).toFixed(2)}%`;
    document.getElementById('game-id').textContent = `ID: ${currentId || ''}`;
    document.getElementById('cover-thumb').src = data.cover || placeholderImage;
    document.getElementById('name').value = data.game.Name || '';
    document.getElementById('summary').value = data.game.Summary || '';
    document.getElementById('first-launch').value = data.game.FirstLaunchDate || '';
    document.getElementById('developers').value = data.game.Developers || '';
    document.getElementById('publishers').value = data.game.Publishers || '';
    setChoices(genresChoices, Array.isArray(data.game.Genres)?data.game.Genres:[]);
    setChoices(modesChoices, Array.isArray(data.game.GameModes)?data.game.GameModes:[]);
    if (data.cover) {
        setImage(data.cover);
        originalImage = data.cover;
    } else {
        clearImage();
        document.getElementById('image').src = placeholderImage;
        originalImage = null;
    }
    currentUpload = null;
    restoreSession();
    if (Array.isArray(data.missing) && data.missing.length) {
        showAlert('Campos vazios: ' + data.missing.join(', '), 'warning');
    }
}

function loadGame() {
    setNavDisabled(true);
    return fetch('api/game')
        .then(r => r.json())
        .then(applyGameData)
        .catch(err => {
            console.error(err.stack || err);
            showAlert('Failed to load game: ' + err.message, 'warning');
        })
        .finally(() => setNavDisabled(false));
}

function saveGame() {
    if (!cropper) { showAlert('Selecione uma imagem', 'warning'); return; }
    const canvas = cropper.getCroppedCanvas({width:1080, height:1080});
    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
    const fields = collectFields();
    setNavDisabled(true);
    const saveBtn = document.getElementById('save');
    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving...';
    fetch('api/save', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({index: currentIndex, id: currentId, fields: fields, image: dataUrl, upload_name: currentUpload})
    })
      .then(async r => {
          const res = await r.json();
          if (!r.ok || res.error) throw new Error(res.error || 'save failed');
          return res;
      })
      .then(() => {
          localStorage.removeItem('session');
          currentUpload = null;
          showAlert('The game was saved.', 'success');
      })
      .catch(err => {
          console.error(err);
          showAlert('Failed to save game: ' + err.message, 'warning');
      })
      .finally(() => {
          setNavDisabled(false);
          saveBtn.disabled = false;
          saveBtn.textContent = 'Save';
      });
}

function skipGame() {
    fetch('api/skip', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({index: currentIndex, upload_name: currentUpload})
    })
      .then(r=>r.json()).then(() => { localStorage.removeItem('session'); loadGame(); })
      .catch(err => {
          console.error(err);
          showAlert('Failed to skip game: ' + err.message, 'warning');
      });
}

function nextGame() {
    if (navigating) return;
    navigating = true;
    setNavDisabled(true);
    fetch('api/next', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({upload_name: currentUpload})
    })
      .then(r=>r.json()).then(data => { localStorage.removeItem('session'); applyGameData(data); })
      .catch(err => {
          console.error(err);
          showAlert('Failed to move to next game: ' + err.message, 'warning');
      })
      .finally(() => {
          navigating = false;
          setNavDisabled(false);
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
          showAlert('Failed to move to previous game: ' + err.message, 'warning');
      })
      .finally(() => {
          navigating = false;
          setNavDisabled(false);
      });
}

function resetFields() {
    fetch(`api/game/${currentIndex}/raw`).then(r=>r.json()).then(data=>{
        document.getElementById('name').value = data.game.Name || '';
        document.getElementById('summary').value = data.game.Summary || '';
        document.getElementById('first-launch').value = data.game.FirstLaunchDate || '';
        document.getElementById('developers').value = data.game.Developers || '';
        document.getElementById('publishers').value = data.game.Publishers || '';
        setChoices(genresChoices, Array.isArray(data.game.Genres)?data.game.Genres:[]);
        setChoices(modesChoices, Array.isArray(data.game.GameModes)?data.game.GameModes:[]);
        if (data.cover) {
            setImage(data.cover);
            originalImage = data.cover;
        } else {
            clearImage();
            document.getElementById('image').src = placeholderImage;
            originalImage = null;
        }
        currentUpload = null;
        saveSession();
    }).catch(err => {
        console.error(err);
        showAlert('Failed to reset fields: ' + err.message, 'warning');
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
            showAlert('Failed to upload image: ' + err.message, 'warning');
        });
});

const genBtn = document.createElement('button');
genBtn.type = 'button';
genBtn.id = 'generate-summary';
genBtn.textContent = 'Gerar Resumo';
document.getElementById('summary').parentNode.appendChild(genBtn);
genBtn.addEventListener('click', generateSummary);

document.getElementById('save').addEventListener('click', saveGame);
document.getElementById('skip').addEventListener('click', skipGame);
document.getElementById('next').addEventListener('click', nextGame);
document.getElementById('previous').addEventListener('click', previousGame);
document.getElementById('reset').addEventListener('click', resetFields);
document.getElementById('revert-image').addEventListener('click', function(){
    fetch(`api/game/${currentIndex}/raw`).then(r=>r.json()).then(data=>{
        if (data.cover) {
            setImage(data.cover);
            originalImage = data.cover;
        } else {
            clearImage();
            document.getElementById('image').src = placeholderImage;
            originalImage = null;
        }
        currentUpload = null;
        saveSession();
    }).catch(err => {
        console.error(err);
        showAlert('Failed to revert image: ' + err.message, 'warning');
    });
});

['name','summary','first-launch','developers','publishers'].forEach(id => {
    document.getElementById(id).addEventListener('change', saveSession);
});

populateSelect('genres', genresList);
populateSelect('modes', modesList);
genresChoices = new Choices('#genres', { removeItemButton: true });
modesChoices = new Choices('#modes', { removeItemButton: true });
genresChoices.passedElement.element.addEventListener('change', saveSession);
modesChoices.passedElement.element.addEventListener('change', saveSession);
loadGame();
