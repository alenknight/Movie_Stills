# Movie Still Finder - Version 1.0

## Initial feature-complete release

- Cross-platform desktop interface for macOS and Windows
- Individual file, multiple-file, and folder-based movie loading
- Movie playback with click-and-drag timeline scrubbing
- Frame rate, resolution, total frame count, and timing details
- Manual marker creation, removal, clearing, and navigation
- Exact first-frame and last-frame markers and extraction
- Frame-based and seconds-based sampling controls
- OpenAI-powered still searches for selected or all loaded videos
- Configurable OpenAI model and confidence settings
- Smart adaptive and evenly spaced GPT review sampling
- Detailed GPT recommendations and follow-up conversations
- Conversion of recommended timestamps into timeline markers
- Batch still extraction with a configurable output folder
- Responsive interface and timestamped success/error messages
- Searchable in-app help describing every control
- Secure API-key handling and GitHub exclusions
- Project documentation and PyCharm setup instructions

## Playback and importing

- Reuse the selected movie decoder during playback and pace previews at the clip frame rate.
- Step one frame with Left or Right and see the current frame number.
- Drop movie files from Finder or File Explorer onto the main window.

## Marker, export, and review fixes

- Preserve manual and review picks across searches and preserve previous results on failure.
- Mark and navigate adjacent frames. Export by frame number with overwrite protection.
- Keep an independent in-memory conversation and verified recommendations for each movie.
