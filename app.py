import os
import json
import uuid
import base64
import io
from typing import Any
from threading import Lock
from tempfile import NamedTemporaryFile

from flask import Flask, request, jsonify, render_template
from PIL import Image, ExifTags
import pandas as pd
from openai import OpenAI

INPUT_XLSX = 'igdb_all_games.xlsx'
PROCESSED_XLSX = 'processed_games.xlsx'
PROGRESS_JSON = 'progress.json'
UPLOAD_DIR = 'uploaded_sources'
PROCESSED_DIR = 'processed_covers'
COVERS_DIR = 'covers_out'

app = Flask(__name__, static_folder='frontend/dist/assets', static_url_path='/assets', template_folder='frontend/dist')

# Configure OpenAI using API key from environment
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', ''))

# Lock to avoid concurrent reads/writes to the processed Excel file
xlsx_lock = Lock()


def safe_write_excel(df: pd.DataFrame, path: str, **kwargs) -> None:
    """Write DataFrame to Excel atomically to avoid corrupt files."""
    dir_name = os.path.dirname(path) or '.'
    tmp = NamedTemporaryFile(delete=False, suffix='.xlsx', dir=dir_name)
    tmp_path = tmp.name
    tmp.close()
    try:
        df.to_excel(tmp_path, **kwargs)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def open_image_auto_rotate(source: Any) -> Image.Image:
    """Open image from path or file-like and auto-rotate using EXIF."""
    img = Image.open(source) if not isinstance(source, Image.Image) else source
    try:
        exif = img._getexif()
        if exif:
            orientation_tag = next(
                k for k, v in ExifTags.TAGS.items() if v == 'Orientation'
            )
            orientation = exif.get(orientation_tag)
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
    except Exception:
        pass
    return img.convert('RGB')


def load_games() -> pd.DataFrame:
    if not os.path.exists(INPUT_XLSX):
        return pd.DataFrame()
    df = pd.read_excel(INPUT_XLSX)
    df = df.dropna(how='all')
    if 'Name' in df.columns:
        df = df[df['Name'].notna()]
    if 'Rating Count' in df.columns:
        df = df.sort_values(by='Rating Count', ascending=False)
    df = df.reset_index(drop=True)
    return df


def load_progress(total_rows: int) -> dict:
    if os.path.exists(PROGRESS_JSON):
        with open(PROGRESS_JSON, 'r') as f:
            return json.load(f)
    progress = {'current_index': 0, 'seq_index': 1, 'skip_queue': []}
    if os.path.exists(PROCESSED_XLSX):
        try:
            with xlsx_lock:
                processed = pd.read_excel(PROCESSED_XLSX)
            progress['seq_index'] = len(processed) + 1
            progress['current_index'] = len(processed)
        except Exception:
            pass
    return progress


def save_progress() -> None:
    with open(PROGRESS_JSON, 'w') as f:
        json.dump(progress, f)


def next_game_index() -> int:
    # decrement countdowns
    for item in progress['skip_queue']:
        item['countdown'] -= 1
    # check for returned skipped items
    for i, item in enumerate(progress['skip_queue']):
        if item['countdown'] <= 0:
            index = item['index']
            del progress['skip_queue'][i]
            save_progress()
            return index
    return progress['current_index']


def find_cover(row: pd.Series) -> str | None:
    url = str(row.get('Large Cover Image (URL)', ''))
    base = os.path.splitext(os.path.basename(url))[0]
    for ext in ('.jpg', '.jpeg', '.png'):
        path = os.path.join(COVERS_DIR, base + ext)
        if os.path.exists(path):
            return path
    return None


def ensure_dirs() -> None:
    for d in (UPLOAD_DIR, PROCESSED_DIR, COVERS_DIR):
        os.makedirs(d, exist_ok=True)


def extract_list(row: pd.Series, keys: list[str]) -> list[str]:
    """Return a list of comma-separated values from the first matching key."""
    for key in keys:
        if key in row.index:
            return [g.strip() for g in str(row.get(key, '')).split(',') if g.strip()]
    return []


def generate_pt_summary(game_name: str) -> str:
    """Generate a simple spoiler-free Portuguese summary for a game by name."""
    if not game_name:
        raise ValueError("game_name is required")
    if not os.environ.get('OPENAI_API_KEY'):
        raise RuntimeError("OPENAI_API_KEY not set")
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {
                'role': 'system',
                'content': (
                    'Você é um assistente que cria sinopses curtas de jogos '
                    'em português do Brasil sem revelar spoilers.'
                ),
            },
            {
                'role': 'user',
                'content': (
                    f"Escreva uma sinopse um pouco mais longa (3 a 5 frases) para o jogo '{game_name}'."
                ),
            },
        ],
        temperature=0.7,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


# initial load
ensure_dirs()
games_df = load_games()
total_games = len(games_df)
progress = load_progress(total_games)
save_progress()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/game')
def api_game():
    index = next_game_index()
    if index >= total_games:
        save_progress()
        return jsonify({'done': True, 'message': 'Todos os jogos foram processados.'})
    row = games_df.iloc[index]
    processed_row = None
    if os.path.exists(PROCESSED_XLSX):
        try:
            with xlsx_lock:
                proc_df = pd.read_excel(PROCESSED_XLSX, dtype=str)
            if 'Source Index' in proc_df.columns:
                match = proc_df[proc_df['Source Index'] == str(index)]
                if not match.empty:
                    processed_row = match.iloc[0]
            else:
                seq_id = f"{progress['seq_index']:07d}"
                match = proc_df[proc_df['ID'] == seq_id]
                if not match.empty:
                    processed_row = match.iloc[0]
        except Exception:
            processed_row = None

    if processed_row is not None:
        cover_path = processed_row.get('Cover Path') or find_cover(row)
    else:
        cover_path = find_cover(row)

    cover_data = None
    if cover_path:
        img = open_image_auto_rotate(cover_path)
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        cover_data = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()

    if processed_row is not None:
        genres = extract_list(processed_row, ['Genres', 'Genre'])
        modes = extract_list(processed_row, ['Game Modes', 'Mode'])
        game_fields = {
            'Name': processed_row.get('Name', ''),
            'Summary': processed_row.get('Summary', ''),
            'FirstLaunchDate': processed_row.get('First Launch Date', ''),
            'Developers': processed_row.get('Developers', ''),
            'Publishers': processed_row.get('Publishers', ''),
            'Genres': genres,
            'GameModes': modes,
        }
    else:
        genres = extract_list(row, ['Genres', 'Genre'])
        modes = extract_list(row, ['Game Modes', 'Mode'])
        game_fields = {
            'Name': row.get('Name', ''),
            'Summary': row.get('Summary', ''),
            'FirstLaunchDate': row.get('First Launch Date', ''),
            'Developers': row.get('Developers', ''),
            'Publishers': row.get('Publishers', ''),
            'Genres': genres,
            'GameModes': modes,
        }

    data = {
        'index': int(index),
        'total': total_games,
        'game': game_fields,
        'cover': cover_data,
        'seq': progress['seq_index'],
    }
    save_progress()
    return jsonify(data)


@app.route('/api/game/<int:index>/raw')
def api_game_raw(index: int):
    if index < 0 or index >= total_games:
        return jsonify({'error': 'invalid index'}), 404
    row = games_df.iloc[index]
    cover_path = find_cover(row)
    cover_data = None
    if cover_path:
        img = open_image_auto_rotate(cover_path)
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        cover_data = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()
    genres = extract_list(row, ['Genres', 'Genre'])
    modes = extract_list(row, ['Game Modes', 'Mode'])
    game_fields = {
        'Name': row.get('Name', ''),
        'Summary': row.get('Summary', ''),
        'FirstLaunchDate': row.get('First Launch Date', ''),
        'Developers': row.get('Developers', ''),
        'Publishers': row.get('Publishers', ''),
        'Genres': genres,
        'GameModes': modes,
    }
    return jsonify({
        'index': int(index),
        'total': total_games,
        'game': game_fields,
        'cover': cover_data,
        'seq': progress['seq_index'],
    })


@app.route('/api/summary', methods=['POST'])
def api_summary():
    data = request.get_json(force=True)
    game_name = data.get('game_name', '')
    try:
        summary_pt = generate_pt_summary(game_name)
        return jsonify({'summary': summary_pt})
    except Exception as e:
        app.logger.exception("Summary generation failed")
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def api_upload():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'no file'}), 400
    img = open_image_auto_rotate(file.stream)
    filename = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(UPLOAD_DIR, filename)
    img.save(path, format='JPEG')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    data = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()
    return jsonify({'filename': filename, 'data': data})


@app.route('/api/save', methods=['POST'])
def api_save():
    data = request.get_json(force=True)
    index = int(data.get('index', 0))
    fields = data.get('fields', {})
    image_b64 = data.get('image')
    upload_name = data.get('upload_name')
    seq_id = f"{progress['seq_index']:07d}"
    df = pd.DataFrame()
    with xlsx_lock:
        if os.path.exists(PROCESSED_XLSX):
            df = pd.read_excel(PROCESSED_XLSX, dtype=str)
            if 'Source Index' in df.columns:
                existing = df[df['Source Index'] == str(index)]
                if not existing.empty:
                    seq_id = existing.iloc[0]['ID']
                    df = df[df['Source Index'] != str(index)]
                else:
                    df = df[df['ID'] != seq_id]
                    progress['seq_index'] += 1
            else:
                df = df[df['ID'] != seq_id]
                progress['seq_index'] += 1
        else:
            progress['seq_index'] += 1

    cover_path = ''
    width = height = 0
    if image_b64:
        header, b64data = image_b64.split(',', 1)
        img = Image.open(io.BytesIO(base64.b64decode(b64data)))
        img = img.convert('RGB')
        if min(img.size) < 1080:
            img = img.resize((1080, 1080))
        else:
            img = img.resize((1080, 1080))
        cover_path = os.path.join(PROCESSED_DIR, f"{seq_id}.jpg")
        img.save(cover_path, format='JPEG', quality=90)
        width, height = img.size

    row = {
        'ID': seq_id,
        'Source Index': str(index),
        'Name': fields.get('Name', ''),
        'Summary': fields.get('Summary', ''),
        'First Launch Date': fields.get('FirstLaunchDate', ''),
        'Developers': fields.get('Developers', ''),
        'Publishers': fields.get('Publishers', ''),
        'Genres': ', '.join(fields.get('Genres', [])),
        'Game Modes': ', '.join(fields.get('GameModes', [])),
        'Cover Path': cover_path,
        'Width': width,
        'Height': height,
    }

    with xlsx_lock:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        safe_write_excel(df, PROCESSED_XLSX, index=False)

    if upload_name:
        up_path = os.path.join(UPLOAD_DIR, upload_name)
        if os.path.exists(up_path):
            os.remove(up_path)
    progress['skip_queue'] = [s for s in progress['skip_queue'] if s['index'] != index]
    save_progress()
    return jsonify({'status': 'ok'})


@app.route('/api/skip', methods=['POST'])
def api_skip():
    data = request.get_json(force=True)
    index = int(data.get('index', 0))
    upload_name = data.get('upload_name')

    progress['skip_queue'] = [s for s in progress['skip_queue'] if s['index'] != index]
    progress['skip_queue'].append({'index': index, 'countdown': 30})
    if index == progress['current_index']:
        progress['current_index'] += 1
    if upload_name:
        up_path = os.path.join(UPLOAD_DIR, upload_name)
        if os.path.exists(up_path):
            os.remove(up_path)
    save_progress()
    return jsonify({'status': 'ok'})


@app.route('/api/next', methods=['POST'])
def api_next():
    data = request.get_json(silent=True) or {}
    upload_name = data.get('upload_name')
    if upload_name:
        up_path = os.path.join(UPLOAD_DIR, upload_name)
        if os.path.exists(up_path):
            os.remove(up_path)
    if progress['current_index'] < total_games:
        progress['current_index'] += 1
    save_progress()
    return jsonify({'status': 'ok'})


@app.route('/api/back', methods=['POST'])
def api_back():
    data = request.get_json(silent=True) or {}
    upload_name = data.get('upload_name')
    if upload_name:
        up_path = os.path.join(UPLOAD_DIR, upload_name)
        if os.path.exists(up_path):
            os.remove(up_path)
    if progress['current_index'] > 0:
        progress['current_index'] -= 1
    save_progress()
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=True)
