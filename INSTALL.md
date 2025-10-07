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
     - `DB_LEGACY_SQLITE=1` forces the historical `processed_games.db` SQLite file if you cannot provision MariaDB yet. Leave this unset for new deployments.
     - `DB_SQLITE_PATH` overrides the SQLite location when `DB_LEGACY_SQLITE=1` (defaults to `processed_games.db` in the current working directory).

3. **Start or provision MariaDB**
   - Point the application at an existing MariaDB 10.6+ instance or launch one locally. A simple Docker Compose service looks like:
     ```yaml
     services:
       mariadb:
         image: mariadb:11
         environment:
           MARIADB_DATABASE: ttgameliste
           MARIADB_USER: ttgameliste
           MARIADB_PASSWORD: example-password
           MARIADB_ROOT_PASSWORD: example-root
         ports:
           - "3306:3306"
         volumes:
           - mariadb-data:/var/lib/mysql
     volumes:
       mariadb-data:
     ```
   - Once MariaDB is accepting connections, confirm the credentials with `mysql`/`mariadb` CLI clients or by running `python -c "from app import db; db.connect().close()"` inside your virtual environment.

4. **(Optional) Add existing artwork**
   - Drop any pre-supplied cover assets into the `covers_out/` directory.
   - The application will create `uploaded_sources/` and `processed_covers/` when it first saves an entry.

5. **Run the app**
   ```bash
   python app.py
   ```
   The server exchanges the Twitch credentials for an IGDB token and downloads the catalogue on startup. Open [http://localhost:5000](http://localhost:5000) and authenticate with the configured password to begin processing games.

6. **(Optional) Realign legacy databases**
   ```bash
   python scripts/resync_db.py
   ```
   Use this helper if you need to bring an existing `processed_games.db` in sync with the freshly fetched IGDB ordering.

Processed data is stored in the MariaDB `processed_games` table by default. Legacy deployments that opt into `DB_LEGACY_SQLITE=1` continue to use `processed_games.db`. Cropped covers are always saved under `processed_covers/`.
