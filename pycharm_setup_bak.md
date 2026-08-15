# PyCharm setup

## Open the project

In PyCharm, choose **File → Open** and select this folder.

## Select the Python environment

1. Open **Settings → Project → Python Interpreter**.
2. Choose **Add Interpreter → Add Local Interpreter → Existing**.
3. Select the interpreter in this project's `.venv` folder:
   - macOS: `.venv/bin/python`
   - Windows: `.venv\Scripts\python.exe`
4. Apply the change.

## Add your OpenAI API key

Recommended PyCharm method:

1. Choose **Run → Edit Configurations**.
2. Add or select a Python configuration.
3. Set the script path to `movie_still_finder.py`.
4. Add an environment variable named `OPENAI_API_KEY` with your key as its value.
5. Run the configuration.

Convenient local-file method:

1. Duplicate `.env.example` and name the copy `.env`.
2. Replace `your_openai_api_key_here` with your key.
3. Run `movie_still_finder.py`.

The `.env` file is listed in `.gitignore`, so it should not be committed. Do not paste a real API key into `.env.example`, the Python source, chat, screenshots, or documentation.

You can also enter a key in the app's masked API-key field. That value exists only in the running app's memory and is not saved when the app closes.
