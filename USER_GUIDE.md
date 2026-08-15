# Movie Still Finder v1.0 User Guide

Movie Still Finder is a cross-platform desktop application for reviewing movie clips, finding prompt-matched moments with OpenAI vision, curating timeline markers, and exporting still images.

> Find the right frame. Mark it. Export it.

## Contents

- [Get the program](#get-the-program)
- [Install on macOS](#install-on-macos)
- [Install on Windows](#install-on-windows)
- [Configure PyCharm](#configure-pycharm)
- [Configure the OpenAI API key](#configure-the-openai-api-key)
- [Quick-start workflow](#quick-start-workflow)
- [Find markers versus GPT review](#find-markers-versus-gpt-review)
- [Sampling and confidence](#sampling-and-confidence)
- [Timeline and markers](#timeline-and-markers)
- [Multiple movies and extraction](#multiple-movies-and-extraction)
- [Complete control reference](#complete-control-reference)
- [Troubleshooting](#troubleshooting)
- [Update the GitHub backup](#update-the-github-backup)

## Get the program

### Download a ZIP

Use this option if you want a copy without Git version history.

1. Open [github.com/alenknight/Movie_Stills](https://github.com/alenknight/Movie_Stills).
2. Click **Code**.
3. Click **Download ZIP**.
4. Open the downloaded ZIP file.
5. Move the `Movie_Stills` folder to your normal project location.

If the repository is private, sign into the GitHub account that has access before downloading it.

### Clone with Git

Use this option if you want version history and easy updates.

```bash
git clone https://github.com/alenknight/Movie_Stills.git
cd Movie_Stills
```

## Install on macOS

Python 3.10 or newer is recommended.

Open Terminal in the project folder and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python movie_still_finder.py
```

If `python3` is unavailable, install a current Python release from [python.org/downloads](https://www.python.org/downloads/), reopen Terminal, and try again.

## Install on Windows

Install Python 3.10 or newer. Enable **Add Python to PATH** during installation.

Open Command Prompt or PowerShell in the project folder and run:

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python movie_still_finder.py
```

If `py` is unavailable, try `python` instead.

## Configure PyCharm

### Open the project

1. Start PyCharm.
2. Choose **File > Open**.
3. Select the `Movie_Stills` folder.
4. Choose **New Window** if another project is already open.

### Select the project interpreter

1. Open **Settings > Project > Python Interpreter**.
2. Choose **Add Interpreter > Add Local Interpreter > Existing**.
3. Select:
   - macOS: `.venv/bin/python`
   - Windows: `.venv\Scripts\python.exe`
4. Apply the change.

### Create the run configuration

1. Choose **Run > Edit Configurations**.
2. Add or select a Python configuration.
3. Name it `Movie Still Finder`.
4. Set the script path to `movie_still_finder.py`.
5. Set the working directory to the main `Movie_Stills` folder.

## Configure the OpenAI API key

Create and manage API keys at [platform.openai.com/api-keys](https://platform.openai.com/api-keys). A complete secret key is shown only when it is created, so save it securely.

API billing is managed separately from a ChatGPT subscription. Review API billing in the [OpenAI Platform](https://platform.openai.com/settings/organization/billing/overview).

### Recommended PyCharm method

1. Choose **Run > Edit Configurations**.
2. Select the `Movie Still Finder` configuration.
3. Open **Environment variables**.
4. Add a variable named `OPENAI_API_KEY`.
5. Paste the key as its value.
6. Leave **Store as project file** unchecked.
7. Apply the configuration and run it.

### Local `.env` method

1. Duplicate `.env.example`.
2. Rename the copy to `.env`.
3. Replace the placeholder with the real key:

```text
OPENAI_API_KEY=your_real_key_here
```

The `.env` file is excluded by `.gitignore` and must never be committed.

### Temporary in-app method

Enter the key in the application's masked API-key field. The value remains in memory for the current run and is not saved when the app closes.

### API-key safety

- Never paste a key into `movie_still_finder.py`.
- Never place a real key in `.env.example`.
- Never include a key in screenshots, chat, documentation, or GitHub.
- If a key is exposed, revoke it and create a replacement.

The app sends sampled JPEG frames and your prompt to the OpenAI API. It does not upload the original movie file.

## Quick-start workflow

1. **Load movies.** Use **Add files** for one or multiple selections, or **Add folder** to scan a folder and its subfolders.
2. **Choose the output folder.** The default is beside the first selected movie. Click **Choose** to change it.
3. **Select a movie.** Click a movie in the list to load its preview, timeline, and clip details.
4. **Describe the still.** Enter a concrete visual prompt.
5. **Choose sampling.** Use seconds for broad scans or frames for precise short clips.
6. **Find markers.** Search the selected video or every loaded video.
7. **Review the markers.** Scrub, play, jump between markers, add missing markers, and remove weak ones.
8. **Extract.** Click **Extract marked stills** to save marked frames from all loaded movies.

Example prompt:

```text
Find visually strong frames that match: close-up face, direct eye contact,
sharp focus, expressive eyes, dark background.
```

## Find markers versus GPT review

### Find markers

Use **Find markers** when you already know what you want.

- Uses the prompt at the bottom of the window.
- Uses the selected seconds/frame interval and confidence threshold.
- Can search one selected movie or every loaded movie.
- Creates timeline markers for matching sampled frames.

This works well for repeatable searches such as close-ups, handstands, smiles, product shots, or specific costumes.

### GPT review selected

Use **GPT review selected** when you want broader recommendations and analysis.

- Reviews one selected movie.
- Uses **Smart adaptive** or **Evenly spaced** sampling.
- Produces a visual summary, candidate timestamps, suggested prompts, and composition notes.
- Supports follow-up questions.
- Can convert recommended timestamps into timeline markers.

This works well for exploration, ranking, creative direction, and discovering moments you did not specify in advance.

Neither mode scans every frame unless the frame interval is `1` and the clip is short enough. Results are limited by the frames sampled and sent to the model.

## Sampling and confidence

### Seconds versus frames

- **Every N seconds** is convenient for broad searches and longer clips.
- **Every N frames** is precise for short clips and brief actions.
- Smaller intervals improve coverage but send more image inputs to the API.

At 24 FPS:

| Frames | Seconds |
|---:|---:|
| 1 | 0.041667 |
| 2 | 0.083333 |
| 3 | 0.125000 |
| 6 | 0.250000 |
| 12 | 0.500000 |

For a 15-20 second clip, try every 6-12 frames first. If a brief moment is missed, reduce the interval and search again.

### Confidence

- A higher confidence value produces fewer, stricter matches.
- A lower confidence value produces more candidates to review manually.
- `0.65` is a reasonable starting point.

## Timeline and markers

- Click or drag the timeline to scrub through the movie.
- Double-click the timeline to add a manual marker.
- Right-click near a marker to remove it.
- **Previous marker** and **Next marker** wrap at the beginning and end.
- **Add marker here** adds a marker at the playhead.
- **Remove nearest marker** removes the closest marker.
- **Clear all markers** clears non-endpoint markers from the selected movie.

When **Include first and last frames** is enabled:

- Both endpoints remain visible on the timeline.
- Previous/Next navigation includes them.
- They are included during extraction.
- Exact frame-number decoding is used for frame `0` and `frame_count - 1`.

## Multiple movies and extraction

### Search all loaded videos

1. Add a folder or multi-select movie files.
2. Enter one search prompt.
3. Choose a sampling interval and confidence threshold.
4. Click **Find markers - all videos in folder**.
5. Select each movie to inspect and edit its marker list.

"All videos" means every movie currently visible in the Movies list, including individually added files. It is not restricted to movies loaded through **Add folder**.

### Extract marked stills

Extraction processes all loaded movies.

- Manual markers are exported.
- Prompt-search markers are exported.
- GPT-recommended markers are exported after they are added to the timeline.
- Enabled first/last-frame markers are exported.
- Output filenames identify the source movie and timestamp.

Before extracting, review each movie, remove weak markers, confirm the output path, and verify the first/last-frame option.

The application reads source movies and writes separate still images. It does not modify or delete the source movies.

## Complete control reference

| ID | Control | What it does |
|---|---|---|
| A | Add files | Adds one or several selected movie files. |
| B | Add folder | Finds supported movies in a folder and its subfolders. |
| C | Clear | Removes loaded movies from the app without deleting source files. |
| D | Output | Displays the still-image destination folder. |
| E | Choose | Selects a different output folder. |
| F | Movies | Lists loaded movies and selects the active preview. |
| G | OpenAI API key | Uses an environment key or a temporary key entered for this run. |
| H | Model | Selects the OpenAI vision model used for searches and reviews. |
| I | Every N seconds | Samples at the entered time interval. |
| J | Every N frames | Samples at the entered frame interval. |
| K | Confidence | Sets the minimum marker-match score. |
| L | Video preview | Displays the selected movie at the playhead. |
| M | Clip details | Shows FPS, resolution, frame count, and seconds per frame. |
| N | Play / Pause | Starts or pauses playback. |
| O | Previous marker | Jumps to the previous marker. |
| P | Next marker | Jumps to the next marker. |
| Q | Timeline | Scrubs the video and displays markers and playhead position. |
| R | Add marker here | Adds a manual marker at the playhead. |
| S | Remove nearest marker | Removes the closest marker. |
| T | Clear all markers | Clears non-endpoint markers from the selected movie. |
| U | Review strategy | Selects Smart adaptive or Evenly spaced GPT review sampling. |
| V | Maximum frames | Limits frames included in a selected-movie GPT review. |
| W | GPT analysis | Displays the visual summary and recommendations. |
| X | Continue conversation | Accepts a follow-up question or instruction. |
| Y | GPT review selected | Reviews the selected movie. |
| Z | Ask follow-up | Continues the current GPT review conversation. |
| AA | Add recommended frames | Converts GPT timestamps into timeline markers. |
| AB | What should ChatGPT find? | Accepts the visual search prompt. |
| AC | Include first and last frames | Keeps endpoint markers and extracts them. |
| AD | Find markers - selected video | Searches only the highlighted movie. |
| AE | Find markers - all videos | Searches every loaded movie. |
| AF | Extract marked stills | Exports markers from all loaded movies. |
| AG | Status bar | Reports progress, success, errors, and timestamps. |
| AH | Help (?) | Opens the searchable in-app control guide. |

## Troubleshooting

### The application will not start

- Confirm the correct `.venv` interpreter is selected.
- Run `python -m pip install -r requirements.txt`.
- Read the first red error in PyCharm's Run pane.

### The API key is missing

- Confirm the variable name is exactly `OPENAI_API_KEY`.
- Run the saved `Movie Still Finder` configuration.
- Leave **Store as project file** unchecked.
- Alternatively, use a local `.env` file or the masked field in the app.

### No markers appear

- Lower the confidence threshold.
- Use a smaller sample interval.
- Make the prompt more specific and visual.
- Confirm internet access and API billing.

### Playback or frame-reading issues

- Try a common MP4 or MOV codec.
- Confirm OpenCV installed successfully.
- Test another clip to isolate a damaged or unusual file.

### The interface is crowded

- Enlarge the application window.
- Drag the pane dividers to resize the Movies, preview, and GPT sections.
- Use the **?** button to open searchable help.

## Update the GitHub backup

After changing the program:

1. Press **Cmd+K** on macOS or **Ctrl+K** on Windows.
2. Review the changed files.
3. Select only the files you intend to upload.
4. Enter a concise commit message.
5. Choose **Commit and Push**.

Never commit:

- `.env`
- `.idea/`
- `venv/` or `.venv/`
- API keys
- Private source movies
- Extracted stills unless you deliberately want them in the repository

Example commit messages:

```text
Add searchable help window
Improve timeline marker controls
Add complete v1 user guide
Fix responsive layout on smaller displays
```

## Additional documentation

- [`README.md`](README.md) - concise project overview
- [`pycharm_setup.md`](pycharm_setup.md) - focused PyCharm setup
- [`Movie_Still_Finder_Help.pdf`](output/pdf/Movie_Still_Finder_Help.pdf) - printable dark user manual
- [OpenAI API keys](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key)
- [OpenAI API billing](https://help.openai.com/en/articles/9039756-managing-billing-settings-on-chatgpt-web-and-platform)
