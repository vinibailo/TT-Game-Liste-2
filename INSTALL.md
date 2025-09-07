# Installation

1. **Install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Prepare data**
   - Place `igdb_all_games.xlsx` in the project root.
   - Put source cover images in the `covers_out/` directory.

3. **Run the app**
   ```bash
   python app.py
   ```
   Then open [http://localhost:5000](http://localhost:5000) in your browser.

Processed data and covers will be written to `processed_games.xlsx` and `processed_covers/`.
