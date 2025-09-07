let cropper = null;
let currentIndex = 0;
let currentUpload = null;
let originalImage = null;
const genresList = ['Ação e Aventura','RPG','Horror de Sobrevivência','Puzzle','Plataforma','Shooter'];
const modesList = ['Single-player','Co-op','PvP'];

function populateSelect(id, options) {
    const sel = document.getElementById(id);
    options.forEach(o => {
        const opt = document.createElement('option');
        opt.value = o; opt.textContent = o;
        sel.appendChild(opt);
    });
}

function setMultiSelect(id, values) {
    const sel = document.getElementById(id);
    Array.from(sel.options).forEach(opt => {
        opt.selected = values && values.includes(opt.value);
    });
}

function collectFields() {
    return {
        Name: document.getElementById('name').value,
        Summary: document.getElementById('summary').value,
        FirstLaunchDate: document.getElementById('first-launch').value,
        Developers: document.getElementById('developers').value,
        Publishers: document.getElementById('publishers').value,
        Genres: Array.from(document.getElementById('genres').selectedOptions).map(o=>o.value),
        GameModes: Array.from(document.getElementById('modes').selectedOptions).map(o=>o.value)
    };
}

function saveSession() {
    const data = {
        index: currentIndex,
        fields: collectFields(),
        image: document.getElementById('image').src,
        upload_name: currentUpload
    };
    localStorage.setItem('session', JSON.stringify(data));
}

function restoreSession() {
    const s = localStorage.getItem('session');
    if (!s) return;
    const data = JSON.parse(s);
    if (data.index !== currentIndex) return;
    document.getElementById('name').value = data.fields.Name;
    document.getElementById('summary').value = data.fields.Summary;
    document.getElementById('first-launch').value = data.fields.FirstLaunchDate;
    document.getElementById('developers').value = data.fields.Developers;
    document.getElementById('publishers').value = data.fields.Publishers;
    setMultiSelect('genres', data.fields.Genres);
    setMultiSelect('modes', data.fields.GameModes);
    if (data.image) setImage(data.image);
    currentUpload = data.upload_name;
}

function updatePreview() {
    if (!cropper) return;
    const canvas = cropper.getCroppedCanvas({width:1080, height:1080});
    document.getElementById('preview').src = canvas.toDataURL('image/jpeg', 0.9);
    saveSession();
}

function setImage(dataUrl) {
    const img = document.getElementById('image');
    img.onload = function(){
        if (cropper) cropper.destroy();
        cropper = new Cropper(img, {aspectRatio:1, viewMode:2, crop:updatePreview, background:false});
        if (Math.min(img.naturalWidth, img.naturalHeight) < 1080) {
            alert('Imagem menor que 1080px será ampliada.');
        }
        updatePreview();
    };
    img.src = dataUrl;
}

function clearImage() {
    if (cropper) { cropper.destroy(); cropper = null; }
    document.getElementById('image').src = '';
    document.getElementById('preview').src = '';
    currentUpload = null;
    saveSession();
}

function loadGame() {
    fetch('/api/game').then(r=>r.json()).then(data => {
        if (data.done) {
            document.body.innerHTML = `<h2>${data.message}</h2>`;
            return;
        }
        currentIndex = data.index;
        document.getElementById('game-name').textContent = data.game.Name || '';
        document.getElementById('caption').textContent = `Jogo ${data.index+1} de ${data.total}`;
        document.getElementById('progress').style.width = `${(data.index+1)/data.total*100}%`;
        document.getElementById('name').value = data.game.Name || '';
        document.getElementById('summary').value = data.game.Summary || '';
        document.getElementById('first-launch').value = data.game.FirstLaunchDate || '';
        document.getElementById('developers').value = data.game.Developers || '';
        document.getElementById('publishers').value = data.game.Publishers || '';
        setMultiSelect('genres', Array.isArray(data.game.Genres)?data.game.Genres:[]);
        setMultiSelect('modes', Array.isArray(data.game.GameModes)?data.game.GameModes:[]);
        if (data.cover) {
            setImage(data.cover);
            originalImage = data.cover;
        } else {
            clearImage();
            originalImage = null;
        }
        currentUpload = null;
        restoreSession();
    });
}

function saveGame() {
    if (!cropper) { alert('Selecione uma imagem'); return; }
    const canvas = cropper.getCroppedCanvas({width:1080, height:1080});
    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
    const fields = collectFields();
    fetch('/api/save', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({index: currentIndex, fields: fields, image: dataUrl, upload_name: currentUpload})})
      .then(r=>r.json()).then(() => { localStorage.removeItem('session'); loadGame(); });
}

function skipGame() {
    fetch('/api/skip', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({index: currentIndex, upload_name: currentUpload})})
      .then(r=>r.json()).then(() => { localStorage.removeItem('session'); loadGame(); });
}

function backGame() {
    fetch('/api/back', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({upload_name: currentUpload})})
      .then(r=>r.json()).then(() => { localStorage.removeItem('session'); loadGame(); });
}

document.getElementById('imageUpload').addEventListener('change', function(){
    const file = this.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    fetch('/api/upload', {method:'POST', body: formData})
        .then(r=>r.json())
        .then(res => { currentUpload = res.filename; setImage(res.data); });
});

document.getElementById('save').addEventListener('click', saveGame);
document.getElementById('skip').addEventListener('click', skipGame);
document.getElementById('back').addEventListener('click', backGame);
document.getElementById('revert-image').addEventListener('click', function(){
    if (originalImage) {
        setImage(originalImage);
    } else {
        clearImage();
    }
    currentUpload = null;
    saveSession();
});

['name','summary','first-launch','developers','publishers','genres','modes'].forEach(id => {
    document.getElementById(id).addEventListener('change', saveSession);
});

populateSelect('genres', genresList);
populateSelect('modes', modesList);
loadGame();
