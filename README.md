# Movie Still Finder

A small macOS/Windows desktop app that samples movie frames, asks OpenAI vision to find frames matching a natural-language prompt, shows matches as yellow markers on a playable timeline, and exports the chosen stills.

## Install

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment:

- macOS: `source .venv/bin/activate`
- Windows: `.venv\Scripts\activate`

Then install and run:

```bash
pip install -r requirements.txt
python movie_still_finder.py
```

For PyCharm-specific instructions, see [pycharm_setup.md](pycharm_setup.md).

Set `OPENAI_API_KEY` in your environment, put it in a local `.env` copied from `.env.example`, or enter it in the app. A key entered in the app stays in memory and is not saved. Never put a real key in the Python source or `.env.example`.

## Workflow

1. Add individual movies, multi-select movies, or add a folder (subfolders are searched too).
2. Describe the visual moments you want to find.
3. Choose the sample interval. A smaller interval finds brief shots more reliably but analyzes more images and costs more.
4. Click **Analyze all movies**. Yellow markers appear on the selected movie's timeline.
5. Curate the result while watching: click the timeline to seek, double-click it to add a marker, and right-click near a marker to remove it. The add/remove buttons do the same thing at the playhead.
6. Click **Extract marked stills**. Manual and AI markers are exported alike. By default, images are saved beside the first selected movie. The first/last-frame option applies even without running analysis.

The app sends sampled JPEG frames—not the original movie—to OpenAI. Match timestamps are therefore accurate to the chosen sampling interval.
