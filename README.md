# Movie Still Finder

A small macOS/Windows desktop app that samples movie frames, asks OpenAI vision to find frames matching a natural-language prompt, shows matches as yellow markers on a playable timeline, and exports the chosen stills.

For complete installation, usage, GUI, troubleshooting, and version-control guidance, see [USER_GUIDE.md](USER_GUIDE.md).

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
4. Under the search prompt, click **Find markers — selected video** for the highlighted movie or **Find markers — all videos in folder** for every loaded movie. Yellow markers appear on the selected movie's timeline.
5. Curate the result while watching: click the timeline to seek, double-click it to add a marker, and right-click near a marker to remove it. The add/remove buttons do the same thing at the playhead. **Clear all markers** removes every non-endpoint marker from the selected movie; first/last markers remain when that option is enabled.
6. Click **Extract marked stills**. Manual and AI markers are exported alike. By default, images are saved beside the first selected movie. The first/last-frame option applies even without running analysis.

Use **Previous marker** and **Next marker** to review matches quickly; navigation wraps at the beginning and end. The model menu offers `gpt-5.4-nano` (lowest cost), `gpt-5.4-mini` (balanced), and `gpt-5.4` (highest quality). You may also type another compatible vision model ID into the model field.

Sampling can be specified either in seconds or frames. The selected movie displays its frame rate, resolution, and seconds per frame; the frame-sampling control also shows the equivalent time interval. At 24 FPS, one frame is approximately 0.041667 seconds, two frames are 0.083333 seconds, and three frames are 0.125 seconds. Very small frame intervals can create thousands of API image inputs, so test them on short clips first.

Long-running progress appears in the bottom status strip. Successful completion is green, errors are red, and final messages include a timestamp instead of opening a completion dialog.

**Find markers — selected video** applies the current still-finding prompt and sampling settings only to the movie highlighted in the Movies list. **Find markers — all videos in folder** does the same marker search for every loaded movie. **GPT review selected** writes a visual summary, recommended still moments, suggested search prompts, and composition notes into the on-screen GPT analysis panel.

GPT review sampling is configurable in the right pane. **Smart adaptive** combines broad timeline coverage with locally detected visual changes/motion; **Evenly spaced** distributes the selected maximum uniformly across the clip. The default maximum is 24 frames, and clips with fewer frames use every available frame. The status strip reports how many frames were actually reviewed and which strategy was used.

Use **GPT review selected** at the lower left of **Continue the GPT conversation** to begin a review. Afterward, type a response or question there and click **Ask follow-up**. Follow-ups continue from the existing review without uploading the sampled frames again.

Click and drag with the left mouse button anywhere on the timeline to scrub through the selected movie. **Add recommended frames to timeline** reads the timestamps in the GPT conversation and turns them into standard timeline markers on the reviewed movie; use **Extract marked stills** afterward to save those frames.

When **Include first and last frames** is enabled, both endpoints appear as visible timeline markers, participate in previous/next marker navigation, and export with `first` and `last` in their filenames. The window sizes itself to the available display area; drag the pane dividers if more room is needed for video or GPT analysis.

Endpoint markers are drawn with their tips exactly at the timeline boundaries. First/last preview and extraction use exact frame-number decoding (`0` and `frame_count - 1`) instead of timestamp seeking, which can round to a nearby frame in compressed movies.

The GPT analysis has a large scrollable pane on the right side of the window. Drag either pane divider to resize the Movies list, video workspace, or analysis area.

The settings column reserves enough width for its labels at larger macOS/Windows display scaling. Timeline help appears beneath its buttons, the GPT conversation buttons share their available width evenly, and the empty-video message remains centered when panes are resized. The video, GPT review, follow-up, and search-prompt fields use compact minimum heights but expand with the window, keeping the bottom prompt, action buttons, and status strip visible when the app is not full screen.

Click the **?** button at the top-right to open a scrollable explanation of every control. Its live search field filters and highlights matching help entries as you type. Click **?** again or **Close help** inside the guide to dismiss it. A matching dark-background PDF guide is included in `output/pdf/Movie_Still_Finder_Help.pdf` with red alphabetical callouts and a full index.

The app sends sampled JPEG frames—not the original movie—to OpenAI. Match timestamps are therefore accurate to the chosen sampling interval.
