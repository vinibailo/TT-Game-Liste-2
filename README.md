# TT-Game-Liste-2

TT-Game-Liste-2 is a password-protected web application for curating game metadata that arrives in bulk from IGDB. It gives an editor-friendly workspace where you can review each record, polish the text fields, generate fresh Portuguese summaries, and produce square 1080×1080 cover art that is ready for storefront ingestion. Every change is written to a local SQLite database so the team can stop at any time and resume later without losing work.

## Feature overview

- **IGDB API ingestion:** Authenticates with Twitch, paginates through IGDB’s `/games` endpoint, sorts the payload by rating count, and keeps every row so editors can process the source data exactly as supplied.【F:app.py†L1515-L1654】
- **Sequential navigation with progress tracking:** Presents one game at a time and tracks the working index, processed sequence, and skipped items. Editors can move forward, go back to the previous entry, or temporarily skip a game while keeping progress metrics accurate.【F:app.py†L150-L357】【F:static/main.js†L94-L212】【F:static/main.js†L245-L353】
- **Metadata editing workspace:** Provides inputs for title, launch date, summary, category, developers, publishers, platforms, genres, and game modes. Chip selectors (Choices.js) make it easy to reuse controlled vocabularies, while an expandable section keeps rarely used metadata out of the way.【F:templates/index.html†L37-L108】【F:static/main.js†L18-L95】
- **1080×1080 cover builder:** Lets the user upload an image, crop it with Cropper.js, and automatically resizes anything smaller than 1080 pixels so the saved JPEG meets platform requirements. A preview updates live while the editor adjusts the crop.【F:templates/index.html†L109-L147】【F:static/main.js†L149-L231】
- **Portuguese summary generation (optional):** When an `OPENAI_API_KEY` is present, the “Gerar Resumo” button asks OpenAI’s Chat Completions API to draft a spoiler-free Brazilian Portuguese synopsis for the current game.【F:app.py†L212-L247】【F:static/main.js†L119-L148】
- **Session resilience:** Draft metadata and image selections are cached in the browser’s `localStorage`, allowing a user to recover unsaved work if they accidentally reload the page.【F:static/main.js†L60-L113】
- **Persistent outputs:** Approved entries land in `processed_games.db`, generated covers are stored as numbered JPEGs inside `processed_covers/`, and source uploads are archived in `uploaded_sources/` until a record is saved or skipped.【F:app.py†L74-L197】【F:app.py†L257-L338】
- **Maintenance tooling:** The `scripts/resync_db.py` helper replays the IGDB fetch to realign the processed database with the source order if files fall out of sync, and `migrate_cover_paths.py` renames exported covers that still use zero-padded filenames.【F:scripts/resync_db.py†L1-L75】【F:migrate_cover_paths.py†L1-L10】

## How the workflow operates

1. **Authentication gate:** The site is protected by a shared password (`APP_PASSWORD`). Editors reach the login form at `/login`, enter the password, and receive a session cookie that unlocks the workspace. You can change the default password via environment variables for production deployments.【F:app.py†L113-L147】【F:templates/login.html†L1-L48】
2. **Game loading:** On startup the server exchanges the configured Twitch credentials for an access token, downloads the IGDB catalogue page by page, and keeps duplicate entries so editors can process the raw feed exactly as supplied. Each row’s index becomes the “Source Index” that drives navigation.【F:app.py†L1515-L1654】
3. **Cover discovery:** When a game already has a local or remote cover URL, the app tries to display it so editors can start from a reasonable crop. Fallback logic downloads remote images on the fly and converts them to inline JPEGs.【F:app.py†L104-L144】
4. **Editing session:** For the active game the UI surfaces editable fields plus a live cropper. Any missing mandatory metadata triggers a warning toast so editors know what to fill in before saving.【F:templates/index.html†L37-L147】【F:static/main.js†L232-L305】
5. **Saving:** On save the client crops the image to 1080×1080, posts the metadata and base64 cover to `/api/save`, and the server writes the record to SQLite while exporting the JPEG. The processed sequence counter increments so IDs remain contiguous.【F:static/main.js†L306-L353】【F:app.py†L248-L338】
6. **Navigation:** Editors can move forward (`Next`), backward (`Previous`), skip troublesome entries, or jump to a specific index via API endpoints. A skip queue automatically revisits skipped games after 30 processed entries so nothing is lost.【F:app.py†L150-L357】【F:static/main.js†L245-L353】
7. **Completion:** When every IGDB row is processed the API responds with a completion message, signalling that the batch is finished.【F:app.py†L318-L338】

## Installing on a clean server

These steps assume a fresh Ubuntu/Debian-like host. Adjust package manager commands to match your distribution.

1. **Install system prerequisites.**
   ```bash
   sudo apt update
   sudo apt install --yes python3 python3-venv python3-pip git
   ```
2. **Clone the repository.**
   ```bash
   git clone <REPO_URL>
   cd TT-Game-Liste-2
   ```
3. **Create and activate a virtual environment.**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
4. **Install Python dependencies.**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. **Seed optional cover assets.** Download or export any existing cover art into the `covers_out/` directory. The server will create other needed folders (`uploaded_sources/`, `processed_covers/`) on first run.【F:app.py†L70-L100】
6. **Configure environment variables.** Create a `.env` file or export the following before launching:
   - `TWITCH_CLIENT_ID` / `TWITCH_CLIENT_SECRET` – required to exchange a Twitch OAuth token for IGDB access.【F:app.py†L1515-L1556】【F:app.py†L1858-L1890】
   - `APP_PASSWORD` – shared password for the login form (defaults to `password`).
   - `APP_SECRET_KEY` – Flask session secret; set a random string in production.
   - `OPENAI_API_KEY` – optional, required only if you want to enable AI-generated summaries.
   - `IGDB_USER_AGENT` – optional, overrides the default User-Agent header if your integration needs a specific contact address.【F:app.py†L146-L147】【F:app.py†L1545-L1562】
7. **Initialize or repair existing data (optional).** If you have a legacy `processed_games.db`, run `python scripts/resync_db.py` to align it with the freshly fetched IGDB source order. To normalise old cover filenames execute `python migrate_cover_paths.py`.
8. **Start the application.**
   ```bash
   python app.py
   ```
   By default Flask listens on `http://127.0.0.1:5000`. Reverse-proxy or bind to 0.0.0.0 if you need remote access.
9. **Log in and begin editing.** Visit `/login`, enter the password you configured, and start processing games. Progress is persisted to `processed_games.db` and can be resumed later.【F:app.py†L248-L338】

## Operational notes

- Keep periodic backups of `processed_games.db` and the `processed_covers/` directory—they contain the authoritative edited content.【F:app.py†L248-L338】
- If the navigation counters ever drift from the IGDB source order, rerun `python scripts/resync_db.py` to resequence IDs and restore alignment.【F:scripts/resync_db.py†L1-L75】
- Enable HTTPS and set a strong `APP_PASSWORD`/`APP_SECRET_KEY` when deploying on the public internet.
- When the AI summary feature is disabled (no OpenAI key), the “Gerar Resumo” button will raise a friendly warning instead of generating text.【F:app.py†L212-L247】【F:static/main.js†L119-L148】

This README complements the short-form instructions in [INSTALL.md](INSTALL.md) with more operational context.
