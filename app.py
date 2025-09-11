import os
import json
import uuid
import base64
import io
import sqlite3
from typing import Any
from threading import Lock
import logging

from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from PIL import Image, ExifTags
import pandas as pd
from openai import OpenAI
from urllib.parse import urlparse
from urllib.request import urlopen

logger = logging.getLogger(__name__)

INPUT_XLSX = 'igdb_all_games.xlsx'
PROCESSED_DB = 'processed_games.db'
UPLOAD_DIR = 'uploaded_sources'
PROCESSED_DIR = 'processed_covers'
COVERS_DIR = 'covers_out'

app = Flask(__name__)
app.secret_key = os.environ.get('APP_SECRET_KEY', 'dev-secret')
APP_PASSWORD = os.environ.get('APP_PASSWORD', 'password')
logging.basicConfig(level=logging.DEBUG)
app.logger.setLevel(logging.DEBUG)
logger.setLevel(logging.DEBUG)


def ensure_processed_db() -> None:
    """Ensure processed games SQLite DB exists, migrating from Excel if needed."""
    if os.path.exists(PROCESSED_DB):
        return
    if not os.path.exists('processed_games.xlsx'):
        logger.info("%s not found, skipping migration", 'processed_games.xlsx')
        return
    try:
        from migrate_to_db import migrate

        migrate()
    except Exception:
        logger.exception("Failed to migrate processed games spreadsheet")


# Configure OpenAI using API key from environment
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', ''))

# SQLite setup for processed games
db_lock = Lock()
ensure_processed_db()
db = sqlite3.connect(PROCESSED_DB, check_same_thread=False)
db.row_factory = sqlite3.Row
with db:
    db.execute(
        '''CREATE TABLE IF NOT EXISTS processed_games (
            "ID" TEXT PRIMARY KEY,
            "Source Index" TEXT UNIQUE,
            "Name" TEXT,
            "Summary" TEXT,
            "First Launch Date" TEXT,
            "Developers" TEXT,
            "Publishers" TEXT,
            "Genres" TEXT,
            "Game Modes" TEXT,
            "Cover Path" TEXT,
            "Width" INTEGER,
            "Height" INTEGER
        )'''
    )
    db.execute(
        '''CREATE TABLE IF NOT EXISTS navigator_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            current_index INTEGER,
            seq_index INTEGER,
            skip_queue TEXT
        )'''
    )


@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.exception("Unhandled exception")
    if request.path.startswith('/api/'):
        return jsonify({'error': 'internal server error'}), 500
    return "Internal Server Error", 500


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
        logger.warning("%s not found", INPUT_XLSX)
        return pd.DataFrame()
    try:
        df = pd.read_excel(INPUT_XLSX)
    except Exception:
        logger.exception("Failed to read %s", INPUT_XLSX)
        return pd.DataFrame()
    df = df.dropna(how='all')
    if 'Name' in df.columns:
        df = df[df['Name'].notna()]
    if 'Rating Count' in df.columns:
        df['Rating Count'] = pd.to_numeric(df['Rating Count'], errors='coerce').fillna(0)
        df = df.sort_values(by='Rating Count', ascending=False, kind='mergesort')
    df = df.drop_duplicates(subset='Name', keep='first')
    df = df.reset_index(drop=True)
    return df




def find_cover(row: pd.Series) -> str | None:
    url = str(row.get('Large Cover Image (URL)', ''))
    if not url:
        return None
    parsed_path = urlparse(url).path
    base = os.path.splitext(os.path.basename(parsed_path))[0]
    for ext in ('.jpg', '.jpeg', '.png'):
        path = os.path.join(COVERS_DIR, base + ext)
        if os.path.exists(path):
            img = open_image_auto_rotate(path)
            buf = io.BytesIO()
            img.save(buf, format='JPEG')
            return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()
    try:
        with urlopen(url) as resp:
            img = open_image_auto_rotate(resp)
            buf = io.BytesIO()
            img.save(buf, format='JPEG')
            return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        app.logger.warning("No cover found for URL %s", url)
    return None


def ensure_dirs() -> None:
    for d in (UPLOAD_DIR, PROCESSED_DIR, COVERS_DIR):
        os.makedirs(d, exist_ok=True)


class GameNavigator:
    """Thread-safe helper to navigate game list and track progress."""

    def __init__(self, total_rows: int):
        self.lock = Lock()
        self.total = total_rows
        self.current_index = 0
        self.seq_index = 1
        self.skip_queue: list[dict[str, int]] = []
        self._load_initial()

    def _load_initial(self) -> None:
        with db_lock:
            cur = db.execute('SELECT current_index, seq_index, skip_queue FROM navigator_state WHERE id=1')
            state_row = cur.fetchone()
            cur = db.execute('SELECT "Source Index", "ID" FROM processed_games')
            rows = cur.fetchall()
        processed = {int(r['Source Index']) for r in rows if str(r['Source Index']).isdigit()}
        max_seq = max((int(r['ID']) for r in rows if str(r['ID']).isdigit()), default=0)
        next_index = next((i for i in range(self.total) if i not in processed), self.total)
        expected_seq = max_seq + 1
        if state_row is not None:
            try:
                file_current = int(state_row['current_index'])
                file_seq = int(state_row['seq_index'])
                file_skip = json.loads(state_row['skip_queue'] or '[]')
                if file_current == next_index and file_seq == expected_seq:
                    self.current_index = file_current
                    self.seq_index = file_seq
                    self.skip_queue = file_skip
                    logger.debug(
                        "Loaded progress: current_index=%s seq_index=%s skip_queue=%s",
                        self.current_index,
                        self.seq_index,
                        self.skip_queue,
                    )
                    return
                logger.warning("Navigator state out of sync with database; rebuilding")
            except Exception as e:
                logger.warning("Failed to load navigator state: %s", e)
        self.current_index = next_index
        self.seq_index = expected_seq
        self.skip_queue = []
        logger.debug(
            "Loaded progress: current_index=%s seq_index=%s skip_queue=%s",
            self.current_index,
            self.seq_index,
            self.skip_queue,
        )
        self._save()

    def _load(self) -> None:
        with db_lock:
            cur = db.execute('SELECT current_index, seq_index, skip_queue FROM navigator_state WHERE id=1')
            state_row = cur.fetchone()
        if state_row is not None:
            try:
                self.current_index = int(state_row['current_index'])
                self.seq_index = int(state_row['seq_index'])
                self.skip_queue = json.loads(state_row['skip_queue'] or '[]')
                logger.debug(
                    "Loaded progress: current_index=%s seq_index=%s skip_queue=%s",
                    self.current_index,
                    self.seq_index,
                    self.skip_queue,
                )
                return
            except Exception as e:
                logger.warning("Failed to load navigator state: %s", e)
        # fallback: rebuild from processed_games
        self._load_initial()

    def _save(self) -> None:
        try:
            with db_lock:
                db.execute(
                    'REPLACE INTO navigator_state (id, current_index, seq_index, skip_queue) VALUES (1, ?, ?, ?)',
                    (
                        self.current_index,
                        self.seq_index,
                        json.dumps(self.skip_queue),
                    ),
                )
                db.commit()
        except Exception as e:
            logger.warning("Failed to save navigator state: %s", e)

    def _process_skip_queue(self) -> None:
        logger.debug(
            "Processing skip queue: index=%s queue=%s",
            self.current_index,
            self.skip_queue,
        )
        for item in self.skip_queue:
            item['countdown'] -= 1
        for i, item in enumerate(self.skip_queue):
            if item['countdown'] <= 0:
                old_index = self.current_index
                self.current_index = item['index']
                del self.skip_queue[i]
                logger.debug(
                    "Skip queue hit: index from %s to %s",
                    old_index,
                    self.current_index,
                )
                break
        logger.debug(
            "After processing skip queue: index=%s queue=%s",
            self.current_index,
            self.skip_queue,
        )

    def current(self) -> int:
        with self.lock:
            self._load()
            self._process_skip_queue()
            self._save()
            return self.current_index

    def next(self) -> int:
        with self.lock:
            self._load()
            before = self.current_index
            logger.debug("next() before skip queue: index=%s", before)
            self._process_skip_queue()
            logger.debug("next() after skip queue: index=%s", self.current_index)
            if self.current_index < self.total:
                self.current_index += 1
            logger.debug("next() after increment: index=%s", self.current_index)
            self._save()
            return self.current_index

    def back(self) -> int:
        with self.lock:
            self._load()
            before = self.current_index
            logger.debug("back() before skip queue: index=%s", before)
            self._process_skip_queue()
            logger.debug("back() after skip queue: index=%s", self.current_index)
            if self.current_index > 0:
                self.current_index -= 1
            logger.debug("back() after decrement: index=%s", self.current_index)
            self._save()
            return self.current_index

    def skip(self, index: int) -> None:
        with self.lock:
            self._load()
            self.skip_queue = [s for s in self.skip_queue if s['index'] != index]
            self.skip_queue.append({'index': index, 'countdown': 30})
            if index == self.current_index:
                self.current_index += 1
            self._save()

    def set_index(self, index: int) -> None:
        with self.lock:
            self._load()
            if 0 <= index <= self.total:
                self.current_index = index
            self._save()


def extract_list(row: pd.Series, keys: list[str]) -> list[str]:
    """Return a list of comma-separated values from the first matching key."""
    for key in keys:
        if key in row.index:
            val = row.get(key, '')
            if pd.isna(val):
                return []
            return [g.strip() for g in str(val).split(',') if g.strip()]
    return []


def get_cell(row: pd.Series, key: str, missing: list[str]) -> str:
    """Return the cell value or an empty string if missing, tracking missing fields."""
    val = row.get(key, '')
    if pd.isna(val) or val is None:
        missing.append(key)
        return ''
    return val


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
navigator = GameNavigator(total_games)

@app.before_request
def require_login():
    if request.endpoint in ('login', 'static'):
        return
    if not session.get('authenticated'):
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form.get('password') == APP_PASSWORD:
            session['authenticated'] = True
            return redirect(url_for('index'))
        error = 'Invalid password'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def index():
    return render_template('index.html', total=total_games)


def build_game_payload(index: int, seq: int) -> dict:
    try:
        row = games_df.iloc[index]
    except Exception:
        raise IndexError('invalid index')
    processed_row = None
    with db_lock:
        cur = db.execute('SELECT * FROM processed_games WHERE "Source Index"=?', (str(index),))
        processed_row = cur.fetchone()

    if processed_row is not None and processed_row['Cover Path']:
        img = open_image_auto_rotate(processed_row['Cover Path'])
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        cover_data = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()
    else:
        cover_data = find_cover(row)

    source_row = pd.Series(dict(processed_row)) if processed_row is not None else row
    genres = extract_list(source_row, ['Genres', 'Genre'])
    modes = extract_list(source_row, ['Game Modes', 'Mode'])
    missing: list[str] = []
    game_fields = {
        'Name': get_cell(source_row, 'Name', missing),
        'Summary': get_cell(source_row, 'Summary', missing),
        'FirstLaunchDate': get_cell(source_row, 'First Launch Date', missing),
        'Developers': get_cell(source_row, 'Developers', missing),
        'Publishers': get_cell(source_row, 'Publishers', missing),
        'Genres': genres,
        'GameModes': modes,
    }

    game_id = processed_row['ID'] if processed_row is not None else f"{seq:07d}"

    return {
        'index': int(index),
        'total': total_games,
        'game': game_fields,
        'cover': cover_data,
        'seq': seq,
        'id': game_id,
        'missing': missing,
    }


@app.route('/api/game')
def api_game():
    try:
        index = navigator.current()
        if index >= total_games:
            return jsonify({'done': True, 'message': 'Todos os jogos foram processados.'})
        data = build_game_payload(index, navigator.seq_index)
        return jsonify(data)
    except Exception as e:
        app.logger.exception("api_game failed")
        return jsonify({'error': str(e)}), 500


@app.route('/api/game/<int:index>/raw')
def api_game_raw(index: int):
    if index < 0 or index >= total_games:
        return jsonify({'error': 'invalid index'}), 404
    try:
        row = games_df.iloc[index]
    except Exception:
        app.logger.exception("api_game_raw failed")
        return jsonify({'error': 'invalid index'}), 404
    cover_data = find_cover(row)
    genres = extract_list(row, ['Genres', 'Genre'])
    modes = extract_list(row, ['Game Modes', 'Mode'])
    dummy: list[str] = []
    game_fields = {
        'Name': get_cell(row, 'Name', dummy),
        'Summary': get_cell(row, 'Summary', dummy),
        'FirstLaunchDate': get_cell(row, 'First Launch Date', dummy),
        'Developers': get_cell(row, 'Developers', dummy),
        'Publishers': get_cell(row, 'Publishers', dummy),
        'Genres': genres,
        'GameModes': modes,
    }
    return jsonify({
        'index': int(index),
        'total': total_games,
        'game': game_fields,
        'cover': cover_data,
        'seq': navigator.seq_index,
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
    expected_id = data.get('id')
    fields = data.get('fields', {})
    image_b64 = data.get('image')
    upload_name = data.get('upload_name')
    if expected_id is None:
        return jsonify({'error': 'missing id'}), 400
    expected_id = str(expected_id)
    try:
        with navigator.lock:
            if index != navigator.current_index:
                return (
                    jsonify(
                        {
                            'error': 'index mismatch',
                            'expected': navigator.current_index,
                            'actual': index,
                        }
                    ),
                    409,
                )
            with db_lock:
                cur = db.execute(
                    'SELECT "ID" FROM processed_games WHERE "Source Index"=?',
                    (str(index),),
                )
                existing = cur.fetchone()
                if existing:
                    existing_id = str(existing['ID'])
                    if existing_id != expected_id:
                        return (
                            jsonify(
                                {
                                    'error': 'id mismatch',
                                    'expected': existing_id,
                                    'actual': expected_id,
                                }
                            ),
                            409,
                        )
                    seq_id = existing_id
                else:
                    seq_id = expected_id
                    navigator.seq_index += 1

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
                "ID": seq_id,
                "Source Index": str(index),
                "Name": fields.get('Name', ''),
                "Summary": fields.get('Summary', ''),
                "First Launch Date": fields.get('FirstLaunchDate', ''),
                "Developers": fields.get('Developers', ''),
                "Publishers": fields.get('Publishers', ''),
                "Genres": ', '.join(fields.get('Genres', [])),
                "Game Modes": ', '.join(fields.get('GameModes', [])),
                "Cover Path": cover_path,
                "Width": width,
                "Height": height,
            }

            with db_lock:
                if existing:
                    db.execute(
                        '''UPDATE processed_games SET
                            "Name"=?, "Summary"=?, "First Launch Date"=?,
                            "Developers"=?, "Publishers"=?, "Genres"=?,
                            "Game Modes"=?, "Cover Path"=?, "Width"=?, "Height"=?
                           WHERE "ID"=?''',
                        (
                            row['Name'], row['Summary'], row['First Launch Date'],
                            row['Developers'], row['Publishers'], row['Genres'],
                            row['Game Modes'], row['Cover Path'], row['Width'], row['Height'],
                            seq_id,
                        ),
                    )
                else:
                    db.execute(
                        '''INSERT INTO processed_games (
                            "ID", "Source Index", "Name", "Summary",
                            "First Launch Date", "Developers", "Publishers",
                            "Genres", "Game Modes", "Cover Path", "Width", "Height"
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (
                            row['ID'], row['Source Index'], row['Name'], row['Summary'],
                            row['First Launch Date'], row['Developers'], row['Publishers'],
                            row['Genres'], row['Game Modes'], row['Cover Path'],
                            row['Width'], row['Height'],
                        ),
                    )
                db.commit()

            if upload_name:
                up_path = os.path.join(UPLOAD_DIR, upload_name)
                if os.path.exists(up_path):
                    os.remove(up_path)
            navigator.skip_queue = [s for s in navigator.skip_queue if s['index'] != index]
            navigator._save()
        return jsonify({'status': 'ok'})
    except Exception as e:
        app.logger.exception("api_save failed")
        return jsonify({'error': str(e)}), 500


@app.route('/api/skip', methods=['POST'])
def api_skip():
    data = request.get_json(force=True)
    index = int(data.get('index', 0))
    upload_name = data.get('upload_name')

    if upload_name:
        up_path = os.path.join(UPLOAD_DIR, upload_name)
        if os.path.exists(up_path):
            os.remove(up_path)
    try:
        navigator.skip(index)
        return jsonify({'status': 'ok'})
    except Exception as e:
        app.logger.exception("api_skip failed")
        return jsonify({'error': str(e)}), 500


@app.route('/api/set_index', methods=['POST'])
def api_set_index():
    data = request.get_json(silent=True) or {}
    index = int(data.get('index', 0))
    upload_name = data.get('upload_name')
    if upload_name:
        up_path = os.path.join(UPLOAD_DIR, upload_name)
        if os.path.exists(up_path):
            os.remove(up_path)
    try:
        navigator.set_index(index)
        return jsonify({'status': 'ok'})
    except Exception as e:
        app.logger.exception("api_set_index failed")
        return jsonify({'error': str(e)}), 500


@app.route('/api/next', methods=['POST'])
def api_next():
    data = request.get_json(silent=True) or {}
    upload_name = data.get('upload_name')
    if upload_name:
        up_path = os.path.join(UPLOAD_DIR, upload_name)
        if os.path.exists(up_path):
            os.remove(up_path)
    try:
        index = navigator.next()
        if index >= total_games:
            return jsonify({'done': True, 'message': 'Todos os jogos foram processados.'})
        payload = build_game_payload(index, navigator.seq_index)
        return jsonify(payload)
    except Exception as e:
        app.logger.exception("api_next failed")
        return jsonify({'error': str(e)}), 500


@app.route('/api/back', methods=['POST'])
def api_back():
    data = request.get_json(silent=True) or {}
    upload_name = data.get('upload_name')
    if upload_name:
        up_path = os.path.join(UPLOAD_DIR, upload_name)
        if os.path.exists(up_path):
            os.remove(up_path)
    try:
        index = navigator.back()
        payload = build_game_payload(index, navigator.seq_index)
        return jsonify(payload)
    except Exception as e:
        app.logger.exception("api_back failed")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
