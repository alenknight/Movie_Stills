#!/usr/bin/env python3
"""Cross-platform GUI for finding and extracting movie stills with OpenAI vision."""

from __future__ import annotations

import base64
import json
import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
try:
    from tkinterdnd2 import COPY, DND_FILES, TkinterDnD
except ImportError:
    TkinterDnD = None  # Existing installations can still load movies with Add files.

WindowRoot = TkinterDnD.Tk if TkinterDnD else tk.Tk


VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".wmv", ".mpg",
    ".mpeg", ".mts", ".m2ts", ".3gp", ".flv",
}

HELP_ITEMS = (
    ("A", "Add files", "Choose movie files, or drop them onto the main window from Finder or File Explorer."),
    ("B", "Add folder", "Find supported movie files inside a folder and its subfolders."),
    ("C", "Clear", "Remove every loaded movie from the app. Source files are never deleted."),
    ("D", "Output", "Shows where extracted still images will be saved."),
    ("E", "Choose", "Select a different output folder."),
    ("F", "Movies", "Lists loaded movies. Click one to show its preview, timeline, and markers."),
    ("G", "OpenAI API key", "Uses OPENAI_API_KEY from .env when available, or accepts a key for this run."),
    ("H", "Model", "Chooses the OpenAI vision model used for marker searches and GPT reviews."),
    ("I", "Every N seconds", "Samples the movie at the entered number of seconds when finding markers."),
    ("J", "Every N frames", "Samples every entered number of frames when finding markers."),
    ("K", "Confidence", "Only keeps GPT marker matches at or above this score."),
    ("L", "Video preview", "Displays the currently selected movie at the playhead position."),
    ("M", "Clip details", "Shows frame rate, resolution, total frame count, and seconds per frame."),
    ("N", "Play / Pause", "Starts or pauses playback. Click the preview and press Left or Right to pause and step one frame."),
    ("O", "Previous marker", "Jumps to the previous marker, wrapping from the first to the last."),
    ("P", "Next marker", "Jumps to the next marker, wrapping from the last to the first."),
    ("Q", "Timeline", "Click or drag to scrub. Yellow triangles are markers; the blue line is the playhead."),
    ("R", "Add marker here", "Adds a manual marker at the current playhead position."),
    ("S", "Remove nearest marker", "Removes the marker closest to the current playhead."),
    ("T", "Clear all markers", "Clears non-endpoint markers from the selected movie."),
    ("U", "Review strategy", "Smart adaptive favors visual changes plus coverage; Evenly spaced uses uniform gaps."),
    ("V", "Maximum frames", "Limits how many sampled frames GPT sees during a selected-movie review."),
    ("W", "GPT analysis", "Displays GPT's visual summary, recommendations, timestamps, and composition notes."),
    ("X", "Continue conversation", "Type a question or instruction about the current GPT review."),
    ("Y", "GPT review selected", "Reviews one selected movie using the review sampling controls."),
    ("Z", "Ask follow-up", "Sends the conversation text as a follow-up to the current review."),
    ("AA", "Add recommended frames", "Converts timestamps from the GPT review into timeline markers."),
    ("AB", "What should ChatGPT find?", "Describe the subjects, poses, framing, action, or visual qualities you want."),
    ("AC", "Include first and last frames", "Keeps endpoint markers visible and includes them during extraction."),
    ("AD", "Find markers - selected video", "Searches sampled frames only in the highlighted movie."),
    ("AE", "Find markers - all videos", "Runs the same marker search on every movie in the Movies list."),
    ("AF", "Extract marked stills", "Exports marked frames from all loaded movies to the output folder."),
    ("AG", "Status bar", "Reports progress, results, errors, and completion timestamps."),
    ("AH", "Help (?)", "Opens this guide. Click ? again or use Close help to hide it."),
)

# Load a developer-local key when present. The .env file is excluded from version control.
load_dotenv()


@dataclass
class Match:
    seconds: float
    confidence: float
    reason: str
    frame_index: int
    source: str


@dataclass
class Review:
    transcript: str = ""
    response_id: str = ""
    samples: dict[float, int] = field(default_factory=dict)
    recommendations: set[int] = field(default_factory=set)
    busy: bool = False
    draft: str = ""


@dataclass(eq=False)
class VideoItem:
    path: Path
    duration: float = 0.0
    fps: float = 0.0
    frame_count: int = 0
    width: int = 0
    height: int = 0
    matches: list[Match] = field(default_factory=list)
    review: Review = field(default_factory=Review)

    def index_at(self, seconds: float) -> int:
        return max(0, min(self.frame_count - 1, round(seconds * self.fps)))

    def marker(self, index: int, source: str, reason: str, confidence: float = 1.0) -> Match:
        return Match(index / self.fps if self.fps else 0, confidence, reason, index, source)


def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def safe_name(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return value[:80] or "still"


def inspect_video(path: Path) -> VideoItem:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open {path.name}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = count / fps if fps > 0 else 0
    capture.release()
    return VideoItem(path=path, duration=duration, fps=fps, frame_count=count, width=width, height=height)


def read_frame_number(path: Path, frame_number: int) -> tuple[bool, object]:
    capture = cv2.VideoCapture(str(path))
    try:
        if capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number):
            ok, frame = capture.read()
            if ok and round(capture.get(cv2.CAP_PROP_POS_FRAMES)) == frame_number + 1:
                return True, frame
        # Some codecs cannot report a reliable seek position. Count from the start.
        capture.release()
        capture = cv2.VideoCapture(str(path))
        for _ in range(frame_number):
            if not capture.grab():
                return False, None
        return capture.read()
    finally:
        capture.release()


def select_review_times(video: VideoItem, maximum: int, mode: str) -> list[float]:
    """Choose broad-coverage frames, optionally favoring visual changes and motion."""
    if video.frame_count <= 0 or video.fps <= 0:
        return [0.0]
    limit = max(2, min(maximum, video.frame_count))
    if video.frame_count <= limit:
        indices = list(range(video.frame_count))
    elif mode == "Evenly spaced":
        indices = sorted({
            round(index * (video.frame_count - 1) / (limit - 1))
            for index in range(limit)
        })
    else:
        scan_count = min(video.frame_count, max(120, limit * 8), 400)
        scan_indices = sorted({
            round(index * (video.frame_count - 1) / (scan_count - 1))
            for index in range(scan_count)
        })
        capture = cv2.VideoCapture(str(video.path))
        scored: list[tuple[float, int]] = []
        previous = None
        for frame_index in scan_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            score = float(cv2.mean(cv2.absdiff(gray, previous))[0]) if previous is not None else 0.0
            scored.append((score, frame_index))
            previous = gray
        capture.release()

        coverage_count = max(4, limit // 2)
        chosen = {
            round(index * (video.frame_count - 1) / (coverage_count - 1))
            for index in range(coverage_count)
        }
        minimum_gap = max(1, video.frame_count // (limit * 4))
        for _score, frame_index in sorted(scored, reverse=True):
            if len(chosen) >= limit:
                break
            if all(abs(frame_index - existing) >= minimum_gap for existing in chosen):
                chosen.add(frame_index)
        if len(chosen) < limit:
            for frame_index in scan_indices:
                chosen.add(frame_index)
                if len(chosen) >= limit:
                    break
        indices = sorted(chosen)[:limit]
    return [frame_index / video.fps for frame_index in indices]


def frame_data_url(frame: object, max_width: int = 768) -> str:
    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        frame = cv2.resize(frame, (max_width, max(1, int(height * scale))))
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise ValueError("Could not encode a sampled frame")
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


class MarkerScale(tk.Canvas):
    def __init__(self, parent: tk.Widget, seek: Callable[[float], None], add: Callable[[], None], remove: Callable[[float], None]):
        super().__init__(parent, height=42, background="#202124", highlightthickness=0)
        self.duration = 0.0
        self.position = 0.0
        self.matches: list[Match] = []
        self.seek = seek
        self.add = add
        self.remove = remove
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<Button-1>", self._clicked)
        self.bind("<B1-Motion>", self._dragged)
        self.bind("<Double-Button-1>", self._double_clicked)
        self.bind("<Button-2>", self._remove_clicked)  # macOS right/control click
        self.bind("<Button-3>", self._remove_clicked)  # Windows right click

    def set_data(self, duration: float, matches: list[Match], position: float = 0) -> None:
        self.duration, self.matches, self.position = duration, matches, position
        self.redraw()

    def set_position(self, position: float) -> None:
        self.position = position
        if self.duration > 0:
            x = position / self.duration * max(1, self.winfo_width())
            self.coords("playhead", x, 0, x, max(1, self.winfo_height()))

    def _clicked(self, event: tk.Event) -> None:
        if self.duration > 0:
            self.seek(max(0.0, min(self.duration, event.x / max(1, self.winfo_width()) * self.duration)))

    def _dragged(self, event: tk.Event) -> None:
        if self.duration > 0:
            self.seek(self._seconds_at(event.x))

    def _seconds_at(self, x: int) -> float:
        return max(0.0, min(self.duration, x / max(1, self.winfo_width()) * self.duration))

    def _double_clicked(self, event: tk.Event) -> None:
        if self.duration > 0:
            seconds = self._seconds_at(event.x)
            self.seek(seconds)
            self.add()

    def _remove_clicked(self, event: tk.Event) -> None:
        if self.duration > 0:
            self.remove(self._seconds_at(event.x))

    def redraw(self) -> None:
        self.delete("all")
        width, height = max(1, self.winfo_width()), max(1, self.winfo_height())
        self.create_rectangle(0, 17, width, 24, fill="#5f6368", outline="")
        if self.duration <= 0:
            return
        for match in self.matches:
            x = match.seconds / self.duration * width
            if match.source == "endpoint" and match.reason == "First frame":
                # Tip is exactly at time zero; the body extends into the canvas.
                self.create_polygon(0, 2, 0, 15, 12, 15, fill="#ffcc00", outline="")
            elif match.source == "endpoint" and match.reason == "Last frame":
                # Tip is exactly at the final frame; the body extends inward.
                self.create_polygon(width, 2, width - 12, 15, width, 15, fill="#ffcc00", outline="")
            else:
                x = max(7, min(width - 7, x))
                self.create_polygon(x, 2, x - 6, 15, x + 6, 15, fill="#ffcc00", outline="")
        x = self.position / self.duration * width
        self.create_line(x, 0, x, height, fill="#52a7ff", width=3, tags="playhead")


class MovieStillFinder(WindowRoot):
    def __init__(self) -> None:
        super().__init__()
        self.title("Movie Still Finder")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        window_width = min(1450, max(900, int(screen_width * .92)))
        window_height = min(820, max(600, int(screen_height * .84)))
        self.geometry(f"{window_width}x{window_height}")
        self.minsize(min(980, int(screen_width * .75)), min(600, int(screen_height * .70)))
        self.videos: list[VideoItem] = []
        self.current: VideoItem | None = None
        self.capture: cv2.VideoCapture | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.playing = False
        self.position = 0.0
        self.help_window: tk.Toplevel | None = None
        self.help_text_widget: tk.Text | None = None
        self.help_search = tk.StringVar()
        self.help_result_count = tk.StringVar()
        self.help_search_trace: str | None = None
        self.play_job: str | None = None
        self.frame_index = -1
        self.next_deadline = 0.0
        self.events: queue.Queue = queue.Queue()
        self.output_dir = tk.StringVar()
        self.api_key = tk.StringVar(value=os.getenv("OPENAI_API_KEY", ""))
        self.model = tk.StringVar(value="gpt-5.4-nano")
        self.interval = tk.DoubleVar(value=10.0)
        self.frame_interval = tk.IntVar(value=3)
        self.sample_mode = tk.StringVar(value="seconds")
        self.threshold = tk.DoubleVar(value=0.65)
        self.include_ends = tk.BooleanVar(value=True)
        self.review_mode = tk.StringVar(value="Smart adaptive")
        self.review_frame_limit = tk.IntVar(value=24)
        self.status = tk.StringVar(value="Add a movie file or folder to begin.")
        self.video_info = tk.StringVar(value="FPS: —  •  Resolution: —  •  Frames: —  •  Seconds/frame: —")
        self.sample_equivalent = tk.StringVar(value="Select a movie to see the frame interval in seconds.")
        self._build_ui()
        self._review_buttons()
        if TkinterDnD:
            self.drop_target_register(DND_FILES)
            self.dnd_bind("<<Drop>>", self._drop_movies)
        else:
            self._set_status("Add files is ready. Install tkinterdnd2 to enable desktop file drops.")
        for key, step in (("Left", -1), ("Right", 1), ("KP_Left", -1), ("KP_Right", 1),
                          ("KP_4", -1), ("KP_6", 1)):
            self.bind(f"<{key}>", lambda event, delta=step: self._step_key(event, delta))
        self.video_canvas.bind("<Button-1>", lambda _: self.video_canvas.focus_set())
        self.frame_interval.trace_add("write", lambda *_args: self._update_sample_equivalent())
        self.sample_mode.trace_add("write", lambda *_args: self._update_sample_equivalent())
        self.after(100, self._poll_events)
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")
        ttk.Button(top, text="Add files…", command=self.add_files).pack(side="left")
        ttk.Button(top, text="Add folder…", command=self.add_folder).pack(side="left", padx=6)
        ttk.Button(top, text="Clear", command=self.clear_files).pack(side="left")
        ttk.Label(top, text="Output:").pack(side="left", padx=(20, 4))
        ttk.Entry(top, textvariable=self.output_dir).pack(side="left", fill="x", expand=True)
        ttk.Button(top, text="Choose…", command=self.choose_output).pack(side="left", padx=(6, 0))
        ttk.Button(top, text="?", width=3, command=self.toggle_help).pack(side="right", padx=(8, 0))

        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10)
        # Keep the settings column wide enough for larger macOS/Windows display
        # scaling. Giving it no expansion weight also stops the center preview
        # from squeezing its labels down to only a few words.
        left = ttk.Frame(body, width=380)
        right = ttk.Frame(body)
        analysis_side = ttk.Frame(body, width=390)
        body.add(left, weight=0)
        body.add(right, weight=4)
        body.add(analysis_side, weight=2)

        ttk.Label(left, text="Movies").pack(anchor="w")
        self.file_list = tk.Listbox(left, exportselection=False)
        self.file_list.pack(fill="both", expand=True, pady=(4, 10))
        self.file_list.bind("<<ListboxSelect>>", self._select_video)

        settings = ttk.LabelFrame(left, text="Analysis", padding=8)
        settings.pack(fill="x")
        ttk.Label(settings, text="OpenAI API key").grid(row=0, column=0, sticky="w")
        ttk.Entry(settings, textvariable=self.api_key, show="•").grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 6))
        ttk.Label(settings, text="Model").grid(row=2, column=0, sticky="w")
        model_picker = ttk.Combobox(
            settings,
            textvariable=self.model,
            values=("gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4"),
        )
        model_picker.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(2, 2))
        ttk.Label(settings, text="Nano: cheapest • Mini: balanced • 5.4: best quality", wraplength=330).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(0, 6)
        )
        ttk.Radiobutton(settings, text="Every N seconds", variable=self.sample_mode, value="seconds").grid(row=5, column=0, sticky="w")
        ttk.Spinbox(settings, from_=.05, to=600, increment=.25, textvariable=self.interval, width=6).grid(row=5, column=1, sticky="e")
        ttk.Radiobutton(settings, text="Every N frames", variable=self.sample_mode, value="frames").grid(row=6, column=0, sticky="w")
        ttk.Spinbox(settings, from_=1, to=10000, textvariable=self.frame_interval, width=6).grid(row=6, column=1, sticky="e")
        ttk.Label(settings, textvariable=self.sample_equivalent, wraplength=330).grid(
            row=7, column=0, columnspan=2, sticky="w", pady=(3, 6)
        )
        ttk.Label(settings, text="Confidence ≥").grid(row=8, column=0, sticky="w")
        ttk.Spinbox(settings, from_=0, to=1, increment=.05, textvariable=self.threshold, width=6).grid(row=8, column=1, sticky="e")
        settings.columnconfigure(0, weight=1)

        # A Canvas has a stable requested size. A Label adopts every new image's
        # dimensions, which can create a resize feedback loop and grow the window.
        self.video_canvas = tk.Canvas(right, width=720, height=260, background="#111111", highlightthickness=0)
        self.video_canvas.pack(fill="both", expand=True)
        self.video_canvas.create_text(360, 130, text="No movie selected", fill="#bbbbbb", tags="empty")
        self.video_canvas.bind("<Configure>", self._resize_video_canvas)
        ttk.Label(right, textvariable=self.video_info, anchor="center").pack(fill="x", pady=(4, 0))
        controls = ttk.Frame(right)
        controls.pack(fill="x", pady=(6, 0))
        self.play_button = ttk.Button(controls, text="▶ Play", command=self.toggle_play)
        self.play_button.pack(side="left")
        ttk.Button(controls, text="◀ Previous marker", command=self.previous_marker).pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="Next marker ▶", command=self.next_marker).pack(side="left", padx=(4, 0))
        self.time_label = ttk.Label(controls, text="00:00:00 / 00:00:00")
        self.time_label.pack(side="left", padx=10)
        self.marker = MarkerScale(right, self.seek, self.add_marker, self.remove_marker_near)
        self.marker.pack(fill="x", pady=4)
        marker_controls = ttk.Frame(right)
        marker_controls.pack(fill="x")
        ttk.Button(marker_controls, text="＋ Add marker here", command=self.add_marker).pack(side="left")
        ttk.Button(marker_controls, text="− Remove nearest marker", command=self.remove_marker_near).pack(side="left", padx=6)
        ttk.Button(marker_controls, text="Clear all markers", command=self.clear_all_markers).pack(side="left")
        ttk.Label(
            right,
            text="Tip: double-click the timeline to add • right-click near a marker to remove",
            anchor="w",
        ).pack(fill="x", pady=(3, 0))

        review_settings = ttk.LabelFrame(analysis_side, text="GPT review sampling", padding=8)
        review_settings.pack(fill="x", pady=(0, 6))
        ttk.Label(review_settings, text="Strategy").pack(side="left")
        ttk.Combobox(
            review_settings,
            textvariable=self.review_mode,
            values=("Smart adaptive", "Evenly spaced"),
            state="readonly",
            width=16,
        ).pack(side="left", padx=(5, 12))
        ttk.Label(review_settings, text="Maximum frames").pack(side="left")
        ttk.Spinbox(
            review_settings,
            from_=2,
            to=200,
            textvariable=self.review_frame_limit,
            width=6,
        ).pack(side="left", padx=(5, 0))

        recommendations = ttk.LabelFrame(analysis_side, text="GPT analysis of selected movie", padding=10)
        recommendations.pack(fill="both", expand=True)
        analysis_scroll = ttk.Scrollbar(recommendations, orient="vertical")
        analysis_scroll.pack(side="right", fill="y")
        self.analysis_text = tk.Text(
            recommendations,
            width=44,
            height=10,
            wrap="word",
            font=("TkDefaultFont", 13),
            padx=10,
            pady=10,
            spacing1=3,
            spacing3=5,
            yscrollcommand=analysis_scroll.set,
        )
        self.analysis_text.pack(side="left", fill="both", expand=True)
        analysis_scroll.configure(command=self.analysis_text.yview)
        self.analysis_text.insert("1.0", "Select a movie, then click “GPT review selected.”")
        self.analysis_text.configure(state="disabled")
        followup_box = ttk.LabelFrame(analysis_side, text="Continue the GPT conversation", padding=8)
        followup_box.pack(fill="x", pady=(6, 0))
        self.followup_text = tk.Text(followup_box, height=2, wrap="word", font=("TkDefaultFont", 12))
        self.followup_text.pack(fill="x")

        followup_actions = ttk.Frame(followup_box)
        followup_actions.pack(fill="x", pady=(6, 0))
        self.recommend_button = ttk.Button(
            followup_actions, text="GPT review selected", command=self.ask_recommendations
        )
        self.recommend_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.followup_button = ttk.Button(followup_actions, text="Ask follow-up", command=self.ask_followup)
        self.followup_button.grid(row=0, column=1, sticky="ew", padx=(3, 0))
        followup_actions.columnconfigure(0, weight=1)
        followup_actions.columnconfigure(1, weight=1)
        self.use_recommendations_button = ttk.Button(
            followup_box,
            text="Add recommended frames to timeline",
            command=self.use_recommended_frames,
        )
        self.use_recommendations_button.pack(fill="x", pady=(5, 0))

        prompt_box = ttk.LabelFrame(self, text="What should ChatGPT find?", padding=8)
        prompt_box.pack(fill="x", padx=10, pady=(8, 4))
        self.prompt = tk.Text(prompt_box, height=2, wrap="word")
        self.prompt.pack(fill="x")
        self.prompt.insert("1.0", "Find visually strong frames that match: ")
        actions = ttk.Frame(prompt_box)
        actions.pack(fill="x", pady=(6, 0))
        ttk.Checkbutton(
            actions,
            text="Include first and last frames",
            variable=self.include_ends,
            command=self.toggle_end_markers,
        ).pack(side="left")
        self.analyze_selected_button = ttk.Button(
            actions, text="Find markers — selected video", command=self.analyze_selected
        )
        self.analyze_selected_button.pack(side="left", padx=(10, 4))
        self.analyze_button = ttk.Button(actions, text="Find markers — all videos in folder", command=self.analyze)
        self.analyze_button.pack(side="left")
        self.extract_button = ttk.Button(actions, text="Extract marked stills", command=self.extract)
        self.extract_button.pack(side="right")

        self.status_bar = tk.Label(
            self,
            textvariable=self.status,
            anchor="w",
            background="#5f6368",
            foreground="white",
            padx=10,
            pady=6,
        )
        self.status_bar.pack(fill="x", padx=10, pady=(2, 8))

    def add_files(self) -> None:
        paths = filedialog.askopenfilenames(title="Choose movie files")
        self._add_paths([Path(p) for p in paths])

    def _drop_movies(self, event) -> str:
        paths = [Path(name) for name in self.tk.splitlist(event.data)]
        movies = [p for p in paths if p.suffix.lower() in VIDEO_EXTENSIONS and p.is_file()]
        if movies:
            self._add_paths(movies)
        else:
            self._set_status("Drop a supported movie file to load it.")
        return COPY

    def toggle_help(self) -> None:
        if self.help_window and self.help_window.winfo_exists():
            self.close_help()
            return

        help_window = tk.Toplevel(self)
        self.help_window = help_window
        help_window.title("Movie Still Finder Help")
        help_window.geometry("860x720")
        help_window.minsize(600, 450)
        help_window.transient(self)
        help_window.configure(background="#202124")
        help_window.protocol("WM_DELETE_WINDOW", self.close_help)

        header = tk.Frame(help_window, background="#202124")
        header.pack(fill="x", padx=16, pady=(14, 8))
        tk.Label(
            header,
            text="Movie Still Finder Help",
            background="#202124",
            foreground="white",
            font=("TkDefaultFont", 18, "bold"),
        ).pack(side="left")
        ttk.Button(header, text="Close help", command=self.close_help).pack(side="right")

        search_row = tk.Frame(help_window, background="#202124")
        search_row.pack(fill="x", padx=16, pady=(0, 8))
        tk.Label(
            search_row,
            text="Search help:",
            background="#202124",
            foreground="white",
        ).pack(side="left", padx=(0, 8))
        search_entry = ttk.Entry(search_row, textvariable=self.help_search)
        search_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(search_row, text="Clear search", command=lambda: self.help_search.set("")).pack(
            side="left", padx=(8, 0)
        )
        tk.Label(
            search_row,
            textvariable=self.help_result_count,
            background="#202124",
            foreground="#b8bcc2",
            width=12,
            anchor="e",
        ).pack(side="right", padx=(8, 0))

        help_text = tk.Text(
            help_window,
            wrap="word",
            background="#171717",
            foreground="#eeeeee",
            insertbackground="white",
            selectbackground="#8b1e2d",
            relief="flat",
            padx=18,
            pady=14,
            font=("TkDefaultFont", 12),
        )
        scrollbar = ttk.Scrollbar(help_window, orient="vertical", command=help_text.yview)
        help_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y", padx=(0, 12), pady=(0, 14))
        help_text.pack(fill="both", expand=True, padx=(16, 0), pady=(0, 14))
        help_text.tag_configure("item", foreground="#ff6666", font=("TkDefaultFont", 12, "bold"))
        help_text.tag_configure("title", foreground="white", font=("TkDefaultFont", 12, "bold"))
        help_text.tag_configure("search_match", background="#8b1e2d", foreground="white")
        self.help_text_widget = help_text
        self.help_search.set("")
        self.help_search_trace = self.help_search.trace_add(
            "write", lambda *_args: self._refresh_help_results()
        )
        self._refresh_help_results()
        help_window.lift()
        help_window.focus_set()
        search_entry.focus_set()

    def _refresh_help_results(self) -> None:
        help_text = self.help_text_widget
        if not help_text or not help_text.winfo_exists():
            return
        query = self.help_search.get().strip()
        query_lower = query.lower()
        matches = [
            item for item in HELP_ITEMS
            if not query_lower or query_lower in " ".join(item).lower()
        ]
        self.help_result_count.set(
            f"{len(matches)} result{'s' if len(matches) != 1 else ''}"
        )
        help_text.configure(state="normal")
        help_text.delete("1.0", "end")
        if matches:
            if not query:
                help_text.insert("end", "Start typing above to filter this guide.\n\n")
            for label, title, description in matches:
                help_text.insert("end", f"{label}  ", "item")
                help_text.insert("end", f"{title}\n", "title")
                help_text.insert("end", f"{description}\n\n")
        else:
            help_text.insert("end", f'No help items match "{query}".\n')
        if query:
            start = "1.0"
            while True:
                found = help_text.search(query, start, stopindex="end", nocase=True)
                if not found:
                    break
                end = f"{found}+{len(query)}c"
                help_text.tag_add("search_match", found, end)
                start = end
        help_text.configure(state="disabled")
        help_text.yview_moveto(0)

    def close_help(self) -> None:
        if self.help_search_trace:
            self.help_search.trace_remove("write", self.help_search_trace)
            self.help_search_trace = None
        if self.help_window and self.help_window.winfo_exists():
            self.help_window.destroy()
        self.help_window = None
        self.help_text_widget = None

    def add_folder(self) -> None:
        folder = filedialog.askdirectory(title="Choose a folder containing movies")
        if folder:
            paths = sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
            self._add_paths(paths)

    def _add_paths(self, paths: list[Path]) -> None:
        existing = {v.path.resolve() for v in self.videos}
        candidates = [p for p in paths if p.suffix.lower() in VIDEO_EXTENSIONS and p.resolve() not in existing]
        if not candidates:
            return
        self._set_status(f"Indexing {len(candidates)} movie(s)…")
        threading.Thread(target=self._index_worker, args=(candidates,), daemon=True).start()

    def _index_worker(self, paths: list[Path]) -> None:
        for path in paths:
            try:
                self.events.put(("indexed", inspect_video(path)))
            except Exception as exc:
                self.events.put(("error", str(exc)))
        self.events.put(("index_done", None))

    def clear_files(self) -> None:
        self._pause()
        self.frame_index = -1
        if self.capture:
            self.capture.release()
        self.capture = None
        self.current = None
        self._show_review()
        self.videos.clear()
        self.file_list.delete(0, "end")
        self.video_canvas.delete("all")
        self.video_canvas.create_text(
            max(1, self.video_canvas.winfo_width()) / 2,
            max(1, self.video_canvas.winfo_height()) / 2,
            text="No movie selected",
            fill="#bbbbbb",
            tags="empty",
        )
        self.marker.set_data(0, [])
        self.video_info.set("FPS: —  •  Resolution: —  •  Frames: —  •  Seconds/frame: —")
        self._update_sample_equivalent()
        self._set_status("File list cleared.")

    def choose_output(self) -> None:
        folder = filedialog.askdirectory(title="Choose output folder")
        if folder:
            self.output_dir.set(folder)

    def _resize_video_canvas(self, event: tk.Event) -> None:
        """Keep the empty-preview message centered as its pane changes size."""
        if self.video_canvas.find_withtag("empty"):
            self.video_canvas.coords("empty", event.width / 2, event.height / 2)

    @staticmethod
    def _set_end_matches(video: VideoItem, enabled: bool) -> None:
        video.matches = [match for match in video.matches if match.source != "endpoint"]
        if enabled and video.frame_count > 0:
            video.matches.append(video.marker(0, "endpoint", "First frame"))
            if video.frame_count > 1:
                video.matches.append(video.marker(video.frame_count - 1, "endpoint", "Last frame"))
        video.matches.sort(key=lambda match: match.frame_index)

    def toggle_end_markers(self) -> None:
        enabled = bool(self.include_ends.get())
        for video in self.videos:
            self._set_end_matches(video, enabled)
        if self.current:
            self.marker.set_data(self.current.duration, self.current.matches, self.position)
        state = "shown" if enabled else "hidden"
        self._set_status(f"First and last frame markers are now {state}.")

    def _select_video(self, _event: object = None) -> None:
        selected = self.file_list.curselection()
        if not selected:
            return
        self._pause()
        self.frame_index = -1
        if self.current:
            self.current.review.draft = self.followup_text.get("1.0", "end-1c")
        self.current = self.videos[selected[0]]
        self._show_review()
        self._set_end_matches(self.current, bool(self.include_ends.get()))
        if self.capture:
            self.capture.release()
        self.capture = cv2.VideoCapture(str(self.current.path))
        self.playing, self.position = False, 0.0
        self.play_button.configure(text="▶ Play")
        self.marker.set_data(self.current.duration, self.current.matches)
        seconds_per_frame = 1 / self.current.fps if self.current.fps > 0 else 0
        self.video_info.set(
            f"FPS: {self.current.fps:.3f}  •  Resolution: {self.current.width}×{self.current.height}  "
            f"•  Frames: {self.current.frame_count:,}  •  Seconds/frame: {seconds_per_frame:.6f}"
        )
        self._update_sample_equivalent()
        self.seek(0)

    def _update_sample_equivalent(self) -> None:
        if not self.current or self.current.fps <= 0:
            self.sample_equivalent.set("Select a movie to see the frame interval in seconds.")
            return
        try:
            frames = max(1, int(self.frame_interval.get()))
        except (tk.TclError, ValueError):
            return
        seconds = frames / self.current.fps
        self.sample_equivalent.set(
            f"{frames} frame{'s' if frames != 1 else ''} = {seconds:.6f} sec at {self.current.fps:.3f} FPS"
        )

    def _set_status(self, message: str, kind: str = "neutral", timestamp: bool = False) -> None:
        colors = {"neutral": "#5f6368", "success": "#218739", "error": "#b3261e"}
        if timestamp:
            message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        self.status.set(message)
        self.status_bar.configure(background=colors.get(kind, colors["neutral"]))

    def seek(self, seconds: float) -> None:
        if self.current:
            self._pause()
            self._display_frame(round(seconds * self.current.fps))

    def _display_frame(self, index: int) -> bool:
        video, capture = self.current, self.capture
        if video is None or capture is None:
            return False
        index = max(0, index)
        if video.frame_count:
            index = min(index, video.frame_count - 1)
        if index == self.frame_index:
            return True
        if round(capture.get(cv2.CAP_PROP_POS_FRAMES)) != index:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                return False
        ok, frame = capture.read()
        if not ok:
            return False
        next_index = round(capture.get(cv2.CAP_PROP_POS_FRAMES))
        self.frame_index = next_index - 1 if next_index > 0 else index
        self.position = self.frame_index / video.fps if video.fps > 0 else 0.0
        self._show_frame(frame)
        self.marker.set_position(self.position)
        self.time_label.configure(text=(
            f"{format_time(self.position)} / {format_time(video.duration)}"
            f"   Frame {self.frame_index + 1:,} of {video.frame_count:,}"
        ))
        return True

    def _step_key(self, event: tk.Event, delta: int) -> str | None:
        if not self.current or event.widget.winfo_toplevel() != self:
            return None
        if event.widget.winfo_class() in {
            "Entry", "TEntry", "Text", "Spinbox", "TSpinbox", "TCombobox", "Scale", "TScale"
        }:
            return None
        self._pause()
        if not self._display_frame(self.frame_index + delta):
            self._set_status("Could not decode the requested frame.", "error", True)
        return "break"

    def add_marker(self) -> None:
        if not self.current:
            return
        video = self.current
        index = self.frame_index
        if index < 0:
            return
        if any(m.frame_index == index and m.source == "manual" for m in video.matches):
            self._set_status(f"Frame {index + 1} already has a manual marker.")
            return
        video.matches = [m for m in video.matches if m.frame_index != index or m.source == "endpoint"]
        video.matches.append(video.marker(index, "manual", "Manual marker"))
        video.matches.sort(key=lambda m: m.frame_index)
        self.marker.set_data(video.duration, video.matches, self.position)
        self._set_status(f"Marked frame {index + 1}.")

    def remove_marker_near(self, seconds: float | None = None) -> None:
        if not self.current or not self.current.matches:
            self._set_status("There are no markers to remove on this movie.")
            return
        target = self.position if seconds is None else seconds
        nearest = min(self.current.matches, key=lambda match: abs(match.seconds - target))
        # A timeline right-click should only remove a marker reasonably near the pointer.
        if seconds is not None:
            tolerance = max(1.0, self.current.duration * .015)
            if abs(nearest.seconds - target) > tolerance:
                self._set_status("Right-click closer to a marker to remove it.")
                return
        self.current.matches = [
            match for match in self.current.matches if match.frame_index != nearest.frame_index
        ]
        self._set_end_matches(self.current, bool(self.include_ends.get()))
        self.marker.set_data(self.current.duration, self.current.matches, self.position)
        self._set_status(f"Removed marker at {format_time(nearest.seconds)}.")

    def clear_all_markers(self) -> None:
        if not self.current:
            self._set_status("Select a movie first.")
            return
        removed = sum(
            match.source != "endpoint" for match in self.current.matches
        )
        self.current.matches = [
            match for match in self.current.matches if match.source == "endpoint"
        ]
        keep_endpoints = bool(self.include_ends.get())
        self._set_end_matches(self.current, keep_endpoints)
        self.marker.set_data(self.current.duration, self.current.matches, self.position)
        endpoint_note = " First/last markers were retained." if keep_endpoints else ""
        self._set_status(
            f"Cleared {removed} marker{'s' if removed != 1 else ''} from "
            f"{self.current.path.name}.{endpoint_note}",
            "success",
            True,
        )

    def _show_frame(self, frame: object) -> None:
        width, height = self.video_canvas.winfo_width(), self.video_canvas.winfo_height()
        h, w = frame.shape[:2]
        scale = min(1.0, max(1, width - 4) / w, max(1, height - 4) / h)
        if scale < 1:
            frame = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                               interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        if self.video_canvas.find_withtag("preview"):
            self.video_canvas.itemconfigure("preview", image=self.photo)
            self.video_canvas.coords("preview", width / 2, height / 2)
        else:
            self.video_canvas.delete("empty")
            self.video_canvas.create_image(width / 2, height / 2, image=self.photo, tags="preview")

    def next_marker(self) -> None:
        if not self.current or not self.current.matches:
            self._set_status("There are no markers on this movie.")
            return
        later = [match for match in self.current.matches if match.frame_index > self.frame_index]
        target = later[0] if later else self.current.matches[0]
        self._pause()
        self._display_frame(target.frame_index)
        self._set_status(f"Marker at {format_time(target.seconds)} — {target.reason}")

    def previous_marker(self) -> None:
        if not self.current or not self.current.matches:
            self._set_status("There are no markers on this movie.")
            return
        earlier = [match for match in self.current.matches if match.frame_index < self.frame_index]
        target = earlier[-1] if earlier else self.current.matches[-1]
        self._pause()
        self._display_frame(target.frame_index)
        self._set_status(f"Marker at {format_time(target.seconds)} — {target.reason}")

    def _pause(self) -> None:
        if self.play_job is not None:
            self.after_cancel(self.play_job)
            self.play_job = None
        self.playing = False
        self.play_button.configure(text="▶ Play")

    def toggle_play(self) -> None:
        if self.playing:
            self._pause()
            return
        if not self.current or self.current.fps <= 0:
            return
        if self.current.frame_count and self.frame_index == self.current.frame_count - 1:
            if not self._display_frame(0):
                return
        self.playing = True
        self.play_button.configure(text="⏸ Pause")
        interval = 1 / self.current.fps
        self.next_deadline = time.monotonic() + interval
        self.play_job = self.after(max(1, round(interval * 1000)), self._play_tick)

    def _play_tick(self) -> None:
        self.play_job = None
        if not self.playing or not self.current:
            return
        target = self.frame_index + 1
        if self.current.frame_count and target >= self.current.frame_count:
            self._pause()
            return
        if not self._display_frame(target):
            self._pause()
            self._set_status("Playback stopped at the last readable frame.")
            return
        interval = 1 / self.current.fps
        now = time.monotonic()
        self.next_deadline = max(now, self.next_deadline + interval)
        self.play_job = self.after(max(1, round((self.next_deadline - now) * 1000)), self._play_tick)

    def analyze(self) -> None:
        self._begin_analysis(list(self.videos))

    def analyze_selected(self) -> None:
        selected = self.file_list.curselection()
        if not selected:
            messagebox.showwarning("No movie selected", "Select one movie in the Movies list first.")
            return
        index = selected[0]
        self._begin_analysis([self.videos[index]])

    def _begin_analysis(self, targets: list[VideoItem]) -> None:
        prompt = self.prompt.get("1.0", "end").strip()
        if not targets or not prompt:
            messagebox.showwarning("Nothing to analyze", "Add at least one movie and enter a search prompt.")
            return
        if not self.api_key.get().strip():
            messagebox.showwarning("API key needed", "Enter an OpenAI API key or set OPENAI_API_KEY.")
            return
        self.analyze_button.configure(state="disabled")
        self.analyze_selected_button.configure(state="disabled")
        scope = "selected movie" if len(targets) == 1 else f"{len(targets)} movies"
        self._set_status(f"Sampling and analyzing {scope}…")
        settings = (
            prompt,
            self.api_key.get().strip(),
            self.model.get().strip(),
            self.sample_mode.get(),
            max(.05, float(self.interval.get())),
            max(1, int(self.frame_interval.get())),
            float(self.threshold.get()),
            targets,
        )
        threading.Thread(target=self._analyze_worker, args=settings, daemon=True).start()

    def _analyze_worker(
        self,
        prompt: str,
        api_key: str,
        model: str,
        sample_mode: str,
        seconds_interval: float,
        frame_interval: int,
        threshold: float,
        targets: list[VideoItem],
    ) -> None:
        try:
            client = OpenAI(api_key=api_key)
            results = []
            for video in targets:
                found = {}
                sampled = False
                if sample_mode == "frames":
                    if video.fps <= 0:
                        raise ValueError(f"Could not determine the frame rate for {video.path.name}")
                    effective_interval = frame_interval / video.fps
                else:
                    effective_interval = seconds_interval
                times = [
                    min(video.duration, n * effective_interval)
                    for n in range(int(video.duration // effective_interval) + 1)
                ]
                if video.duration and (not times or times[-1] < video.duration - .25):
                    times.append(max(0, video.duration - .1))
                for start in range(0, len(times), 12):
                    batch = times[start:start + 12]
                    content: list[dict] = [{
                        "type": "input_text",
                        "text": (
                            "You are selecting movie stills. Evaluate each timestamped image against this request: "
                            f"{prompt}\nReturn JSON only in this exact form: "
                            '{"matches":[{"index":0,"confidence":0.0,"reason":"short explanation"}]}. '
                            "Include only genuine visual matches. Confidence must be 0 to 1. Index is the supplied frame index."
                        ),
                    }]
                    mapping: dict[int, int] = {}
                    for local_index, seconds in enumerate(batch):
                        index = video.index_at(seconds)
                        ok, frame = read_frame_number(video.path, index)
                        if not ok:
                            continue
                        sampled = True
                        mapping[local_index] = index
                        seconds = index / video.fps if video.fps else 0
                        content.append({"type": "input_text", "text": f"Frame index {local_index}, timestamp {seconds:.3f} seconds:"})
                        content.append({"type": "input_image", "image_url": frame_data_url(frame), "detail": "low"})
                    if not mapping:
                        continue
                    response = client.responses.create(
                        model=model,
                        input=[{"role": "user", "content": content}],
                        text={"format": {"type": "json_object"}},
                    )
                    result = json.loads(response.output_text)
                    for item in result.get("matches", []):
                        index = int(item.get("index", -1))
                        confidence = float(item.get("confidence", 0))
                        if index in mapping and confidence >= threshold:
                            frame_index = mapping[index]
                            found[frame_index] = video.marker(frame_index, "search", str(item.get("reason", "Match")), confidence)
                    self.events.put(("progress", f"Analyzed {video.path.name}: {min(start + 12, len(times))}/{len(times)} samples"))
                if not sampled:
                    raise ValueError(f"Could not sample {video.path.name}")
                results.append((video, found))
            self.events.put(("analyze_done", results))
        except Exception as exc:
            self.events.put(("analyze_failed", f"Search failed ({type(exc).__name__}). Your markers were kept."))

    def ask_recommendations(self) -> None:
        selected = self.file_list.curselection()
        if not selected:
            messagebox.showwarning("No movie selected", "Select one movie in the Movies list first.")
            return
        if not self.api_key.get().strip():
            messagebox.showwarning("API key needed", "Enter an OpenAI API key or set OPENAI_API_KEY.")
            return
        video = self.videos[selected[0]]
        goal = self.prompt.get("1.0", "end").strip()
        if video.review.busy:
            return
        video.review.busy = True
        self._review_buttons()
        self._set_status(f"Asking GPT to review {video.path.name}…")
        threading.Thread(
            target=self._recommendations_worker,
            args=(
                video,
                goal,
                self.api_key.get().strip(),
                self.model.get().strip(),
                self.review_mode.get(),
                max(2, int(self.review_frame_limit.get())),
            ),
            daemon=True,
        ).start()

    def _recommendations_worker(
        self,
        video: VideoItem,
        goal: str,
        api_key: str,
        model: str,
        review_mode: str,
        review_frame_limit: int,
    ) -> None:
        try:
            client = OpenAI(api_key=api_key)
            times = select_review_times(video, review_frame_limit, review_mode)
            content: list[dict] = [{
                "type": "input_text",
                "text": (
                    f"Review these {len(times)} chronological frames selected using the {review_mode} strategy "
                    "from one short movie clip. Provide: "
                    "(1) a concise visual summary, (2) the strongest candidate still moments with their supplied "
                    "timestamps and why they work, (3) 3-5 useful search prompts for finding stills in this clip, "
                    "and (4) any visible technical or composition notes. Do not claim to hear audio or see frames "
                    "that were not supplied. Format the answer as easy-to-read plain text with short headings and "
                    "blank lines; do not use Markdown symbols such as #, **, or --- . Start every recommended still "
                    "line with MARKER: followed by its timestamp in seconds, for example MARKER: 4.250s.\n\n"
                    "The user's current still-finding goal is:\n" + (goal or "Not specified")
                ),
            }]
            samples = {}
            for index, seconds in enumerate(times):
                frame_index = video.index_at(seconds)
                ok, frame = read_frame_number(video.path, frame_index)
                if not ok:
                    continue
                samples[round(seconds, 3)] = frame_index
                content.append({
                    "type": "input_text",
                    "text": f"Frame {index + 1}, timestamp {seconds:.3f} seconds:",
                })
                content.append({"type": "input_image", "image_url": frame_data_url(frame), "detail": "low"})
            if not samples:
                raise ValueError(f"Could not sample frames from {video.path.name}")
            response = client.responses.create(
                model=model,
                input=[{"role": "user", "content": content}],
            )
            self.events.put((
                "recommendations_done",
                (video, response.output_text, response.id, samples, review_mode),
            ))
        except Exception as exc:
            self.events.put(("review_failed", (video, type(exc).__name__)))

    def ask_followup(self) -> None:
        question = self.followup_text.get("1.0", "end").strip()
        video = self.current
        if not video or not video.review.response_id:
            messagebox.showwarning("Review a movie first", "Click “GPT review selected” before asking a follow-up.")
            return
        if not question:
            messagebox.showwarning("Enter a question", "Type a follow-up question first.")
            return
        if not self.api_key.get().strip():
            messagebox.showwarning("API key needed", "Enter an OpenAI API key or set OPENAI_API_KEY.")
            return
        if video.review.busy:
            return
        video.review.busy = True
        self._review_buttons()
        self._set_status(f"Asking a follow-up about {video.path.name}…")
        threading.Thread(
            target=self._followup_worker,
            args=(
                video, question,
                video.review.response_id,
                self.api_key.get().strip(),
                self.model.get().strip(),
            ),
            daemon=True,
        ).start()

    def _followup_worker(self, video: VideoItem, question: str, previous_response_id: str, api_key: str, model: str) -> None:
        try:
            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model=model,
                previous_response_id=previous_response_id,
                instructions="Use only previously supplied timestamps. Start each recommended frame line with MARKER: and its supplied timestamp in seconds.",
                input=[{"role": "user", "content": [{"type": "input_text", "text": question}]}],
            )
            self.events.put(("followup_done", (video, question, response.output_text, response.id)))
        except Exception as exc:
            self.events.put(("review_failed", (video, type(exc).__name__)))

    def _review_buttons(self) -> None:
        review = self.current.review if self.current else None
        ready = review is not None and not review.busy
        self.recommend_button.configure(state="normal" if ready else "disabled")
        self.followup_button.configure(state="normal" if ready and review.response_id else "disabled")
        self.use_recommendations_button.configure(state="normal" if ready and review.recommendations else "disabled")

    def _show_review(self) -> None:
        review = self.current.review if self.current else None
        self.analysis_text.configure(state="normal")
        self.analysis_text.delete("1.0", "end")
        self.analysis_text.insert("1.0", review.transcript if review and review.transcript else "Select a movie and click GPT review selected.")
        self.analysis_text.configure(state="disabled")
        self.analysis_text.see("end")
        self.followup_text.delete("1.0", "end")
        if review:
            self.followup_text.insert("1.0", review.draft)
        self._review_buttons()

    @staticmethod
    def _record_recommendations(review: Review, answer: str) -> None:
        for value in re.findall(r"(?im)^\s*(?:[-*]\s*)?MARKER\s*:\s*(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)\b", answer):
            index = review.samples.get(float(value))
            if index is not None:
                review.recommendations.add(index)

    def use_recommended_frames(self) -> None:
        video = self.current
        if not video or video.review.busy:
            return
        added = 0
        for index in sorted(video.review.recommendations):
            existing = [m for m in video.matches if m.frame_index == index]
            if any(m.source in {"manual", "review"} for m in existing):
                continue
            video.matches = [m for m in video.matches if m.frame_index != index or m.source == "endpoint"]
            video.matches.append(video.marker(index, "review", "GPT recommended frame"))
            added += 1
        video.matches.sort(key=lambda m: m.frame_index)
        self.marker.set_data(video.duration, video.matches, self.position)
        self._set_status(f"Added {added} reviewed frame(s) to {video.path.name}.", "success", True)

    def extract(self) -> None:
        if not self.videos:
            return
        output = Path(self.output_dir.get().strip()) if self.output_dir.get().strip() else self.videos[0].path.parent
        self.extract_button.configure(state="disabled")
        self._set_status("Extracting stills…")
        include_ends = bool(self.include_ends.get())
        jobs = []
        for video in self.videos:
            self._set_end_matches(video, include_ends)
            unique = {m.frame_index: m for m in video.matches}
            unique.update((m.frame_index, m) for m in video.matches if m.source == "endpoint")
            jobs.append((video.path, tuple(unique.values())))
        threading.Thread(target=self._extract_worker, args=(output, jobs), daemon=True).start()

    def _extract_worker(self, output: Path, jobs: list[tuple[Path, tuple[Match, ...]]]) -> None:
        count = 0
        try:
            output.mkdir(parents=True, exist_ok=True)
            for path, matches in jobs:
                for match in matches:
                    ok, frame = read_frame_number(path, match.frame_index)
                    if not ok:
                        raise OSError(f"Could not read frame {match.frame_index + 1} of {path.name}")
                    ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if not ok:
                        raise OSError(f"Could not encode frame {match.frame_index + 1}")
                    label = match.reason.split()[0].lower() if match.source == "endpoint" else match.source
                    stem = f"{safe_name(path.stem)}_frame_{match.frame_index + 1:08d}_{label}"
                    suffix = 1
                    while True:
                        name = f"{stem}.jpg" if suffix == 1 else f"{stem}_{suffix}.jpg"
                        try:
                            stream = (output / name).open("xb")
                        except FileExistsError:
                            suffix += 1
                            continue
                        try:
                            with stream:
                                stream.write(data.tobytes())
                        except OSError:
                            (output / name).unlink(missing_ok=True)
                            raise
                        break
                    count += 1
            self.events.put(("extract_done", (count, output)))
        except Exception as exc:
            self.events.put(("extract_failed", f"Saved {count} still(s) before the error. {exc}"))

    def _poll_events(self) -> None:
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "indexed":
                    self.videos.append(value)
                    self.file_list.insert("end", f"{value.path.name}  ({format_time(value.duration)})")
                    if len(self.videos) == 1:
                        self.file_list.selection_set(0)
                        self._select_video()
                        self.output_dir.set(str(value.path.parent))
                elif kind == "index_done":
                    self._set_status(f"Indexed {len(self.videos)} movie(s).", "success", timestamp=True)
                elif kind in {"error", "analyze_failed", "extract_failed"}:
                    if kind == "analyze_failed":
                        self.analyze_button.configure(state="normal")
                        self.analyze_selected_button.configure(state="normal")
                    if kind == "extract_failed":
                        self.extract_button.configure(state="normal")
                    self._set_status(f"Error: {value}", "error", timestamp=True)
                elif kind == "progress":
                    self._set_status(value)
                elif kind == "analyze_done":
                    total = 0
                    for video, found in value:
                        if video not in self.videos:
                            continue
                        kept = [m for m in video.matches if m.source in {"manual", "review"}]
                        occupied = {m.frame_index for m in kept}
                        video.matches = kept + [m for i, m in found.items() if i not in occupied]
                        self._set_end_matches(video, bool(self.include_ends.get()))
                        total += len(found)
                        if self.current is video:
                            self.marker.set_data(video.duration, video.matches, self.position)
                    self.analyze_button.configure(state="normal")
                    self.analyze_selected_button.configure(state="normal")
                    self._set_status(f"Search complete. Found {total} matching frame(s).", "success", True)
                elif kind in {"recommendations_done", "followup_done", "review_failed"}:
                    video = value[0]
                    if video not in self.videos:
                        continue
                    review = video.review
                    review.busy = False
                    if kind == "review_failed":
                        if self.current is video:
                            self._review_buttons()
                            self._set_status(f"GPT request failed for {video.path.name} ({value[1]}).", "error", True)
                        continue
                    if self.current is video:
                        review.draft = self.followup_text.get("1.0", "end-1c")
                    if kind == "recommendations_done":
                        _, answer, review.response_id, review.samples, mode = value
                        review.transcript = answer
                        review.recommendations.clear()
                        message = f"Reviewed {len(review.samples)} frames of {video.path.name} ({mode})."
                    else:
                        _, question, answer, review.response_id = value
                        review.transcript += f"\n\nYOU\n{question}\n\nGPT\n{answer}"
                        if review.draft.strip() == question:
                            review.draft = ""
                        message = f"Follow-up complete for {video.path.name}."
                    self._record_recommendations(review, answer)
                    if self.current is video:
                        self._show_review()
                        self._set_status(message, "success", True)
                elif kind == "extract_done":
                    count, output = value
                    self.extract_button.configure(state="normal")
                    self._set_status(
                        f"Success: saved {count} still(s) to {output}",
                        "success",
                        timestamp=True,
                    )
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _close(self) -> None:
        self._pause()
        if self.capture:
            self.capture.release()
        self.destroy()


if __name__ == "__main__":
    MovieStillFinder().mainloop()
