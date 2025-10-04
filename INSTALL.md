# Installation

1. **Create a virtual environment and install requirements**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Provide IGDB credentials and app secrets**
   - Register a Twitch Developer application and copy its Client ID and Client Secret.
   - Export `TWITCH_CLIENT_ID` and `TWITCH_CLIENT_SECRET`, or place them in a `.env` file alongside `APP_PASSWORD`/`APP_SECRET_KEY`.
   - Optional variables:
     - `OPENAI_API_KEY` enables automated Portuguese summaries.
     - `IGDB_USER_AGENT` customises the User-Agent header sent to IGDB.
     - `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` configure the MariaDB connection. The app uses SQLAlchemy URLs (`mysql+pymysql://…`) and requires the pure-Python [`PyMySQL`](https://pypi.org/project/PyMySQL/) driver included in `requirements.txt`.
     - `DB_SSL_CA` provides an SSL CA bundle when the database enforces TLS validation.
     - `DB_CONNECT_TIMEOUT` / `DB_READ_TIMEOUT` tune connection/read timeouts in seconds.
     - `DB_LEGACY_SQLITE=1` forces the historical `processed_games.db` SQLite file if you cannot provision MariaDB yet.
     - `DB_SQLITE_PATH` overrides the SQLite location when `DB_LEGACY_SQLITE=1` (defaults to `processed_games.db` in the current working directory).

3. **(Optional) Add existing artwork**
   - Drop any pre-supplied cover assets into the `covers_out/` directory.
   - The application will create `uploaded_sources/` and `processed_covers/` when it first saves an entry.

4. **Run the app**
   ```bash
   python app.py
   ```
   The server exchanges the Twitch credentials for an IGDB token and downloads the catalogue on startup. Open [http://localhost:5000](http://localhost:5000) and authenticate with the configured password to begin processing games.

5. **(Optional) Realign legacy databases**
   ```bash
   python scripts/resync_db.py
   ```
   Use this helper if you need to bring an existing `processed_games.db` in sync with the freshly fetched IGDB ordering.

Processed data is stored in `processed_games.db`, and cropped covers are saved under `processed_covers/`.
