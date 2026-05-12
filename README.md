# Mock Interview Reel Generator

Automatically builds a short-form vertical video reel for a given difficulty level.
The output is a single `.mp4` ready to post — no editing required.

---

## What it produces

```
Intro scene  →  Difficulty reveal  →  Question slide-in
```

| Segment | What you see | What you hear |
|---|---|---|
| **Intro** | A random background video with word-by-word subtitles | A random audio file from `audios/` |
| **Difficulty** | The difficulty image filling the frame | `sfx/riser.mp3` |
| **Question** | The question image sliding in from the right, over the difficulty background | Silence |

---

## Requirements

```bash
pip install moviepy pillow openai-whisper numpy
```

> MoviePy **2.x** is required. Version 1.x has a different API and will not work.

---

## How to run

```bash
python main.py <difficulty>
```

**Examples:**

```bash
python main.py easy
python main.py medium
python main.py hard
```

The finished video is saved to:

```
output/reel_<difficulty>.mp4
```

---

## Folder structure

Every folder below must exist before you run the program.
Create them manually if they are missing.

```
mock-interview/
│
├── main.py                  # Entry point — parses difficulty and calls pipeline.run()
├── pipeline.py              # Orchestrates the full scene build + export
├── config.py                # All paths and settings in one place
│
├── core/
│   ├── scenes.py            # Scene builders (intro, difficulty, question)
│   ├── subtitles.py         # Word-by-word subtitle clip renderer
│   ├── transcribe.py        # Whisper transcription wrapper
│   └── video.py             # Video helpers (9:16 crop, looping, cover crop)
│
├── utils.py                 # Shared utilities (print_step, pick_random_file)
│
├── videos/                  # ← DROP background video files here
│   └── *.mp4 / *.mov / *.avi / *.mkv
│
├── audios/                  # ← DROP voiceover audio files here
│   └── *.mp3 / *.wav / *.m4a / *.aac
│
├── difficulty/              # ← One PNG per difficulty level
│   ├── easy.png
│   ├── medium.png
│   └── hard.png
│
├── questions/               # ← Question image(s) — see naming rules below
│   └── question.png
│
├── sfx/
│   └── riser.mp3            # ← Required. Plays during the difficulty reveal.
│
└── output/                  # Auto-populated. Finished reels saved here.
```

---

## What to put in each folder

### `videos/`
Any number of background video clips. One is picked at random each run.
Supported formats: `.mp4`, `.mov`, `.avi`, `.mkv`

The video is automatically cropped and reframed to **9:16** — landscape or square
clips both work fine.

---

### `audios/`
Your voiceover or commentary audio files. One is picked at random each run.
Supported formats: `.mp3`, `.wav`, `.m4a`, `.aac`

The video is looped to match the audio's length, and Whisper transcribes
the audio automatically to generate the word-by-word subtitles.

---

### `difficulty/`
One PNG per difficulty level you intend to use, named exactly after the
difficulty string you pass on the command line.

```
difficulty/easy.png
difficulty/medium.png
difficulty/hard.png
```

The image is cover-cropped to fill the full **9:16** frame.

---

### `questions/`
The image that slides in during the question segment.

The program checks for a match in this order — first one found wins:

| Priority | Filename | Use case |
|---|---|---|
| 1 | `questions/<difficulty>.png` | Different question per difficulty |
| 2 | `questions/<difficulty>.jpg` | Same, JPEG variant |
| 3 | `questions/question.png` | One image used for all difficulties |
| 4 | `questions/question.jpg` | Same, JPEG variant |

**The simplest setup:** drop a single `question.png` into the folder and it
will be used regardless of which difficulty you run.

The image is automatically scaled to fill the full **9:16** frame.

---

### `sfx/riser.mp3`
A single audio file that plays under the difficulty reveal.
If it is longer than `DIFFICULTY_DURATION` (set in `config.py`), it is
trimmed automatically.

---

## config.py — key settings

Open `config.py` to change any of these without touching pipeline code.

| Setting | Default | What it controls |
|---|---|---|
| `OUT_W` / `OUT_H` | `1080` / `1920` | Output frame size (9:16) |
| `OUTPUT_FPS` | `30` | Frames per second of the exported video |
| `OUTPUT_CODEC` | `libx264` | Video codec passed to FFMPEG |
| `OUTPUT_PRESET` | `fast` | FFMPEG encoding speed/quality trade-off |
| `DIFFICULTY_DURATION` | e.g. `3.0` | Max seconds the difficulty reveal is shown |
| `VIDEOS_DIR` | `Path("videos")` | Where background videos are read from |
| `AUDIOS_DIR` | `Path("audios")` | Where voiceover audio is read from |
| `DIFFICULTY_DIR` | `Path("difficulty")` | Where difficulty PNGs are read from |
| `SFX_DIR` | `Path("sfx")` | Where `riser.mp3` lives |
| `OUTPUT_DIR` | `Path("output")` | Where finished reels are written |

---

## Common errors and fixes

**`FileNotFoundError: No question image found`**
→ Add a `question.png` to the `questions/` folder.

**`FileNotFoundError: Difficulty image not found`**
→ Make sure `difficulty/<difficulty>.png` exists and the filename matches
the argument you passed exactly (all lowercase).

**`FileNotFoundError: Riser SFX not found`**
→ Make sure `sfx/riser.mp3` exists.

**`No files found in <folder>`**
→ The `videos/` or `audios/` folder is empty. Drop at least one file in each.

**`WinError 6: The handle is invalid`** *(Windows only, appears after export)*
→ This is a known MoviePy/Windows cosmetic warning that appears after the
video has already been saved successfully. It does not affect the output.
It has been suppressed in the current version by explicitly closing clips
after export.

---

## Pipeline flow (for reference)

```
main.py
  └── pipeline.run(difficulty)
        ├── build_intro_scene()
        │     ├── pick random video from videos/
        │     ├── pick random audio from audios/
        │     ├── transcribe audio → word timestamps (Whisper)
        │     ├── crop + loop video to audio length
        │     └── composite video + subtitle clips
        │
        ├── build_difficulty_scene(difficulty)
        │     ├── load difficulty/<difficulty>.png
        │     ├── cover-crop to 9:16
        │     └── hold image for riser audio duration
        │
        ├── build_question_scene(image_path, bg_clip)
        │     ├── freeze last frame of difficulty clip as background
        │     ├── load + scale question image to 9:16
        │     └── animate slide-in from right (ease-out quad, 0.45 s)
        │
        ├── _stitch()  →  intro + difficulty + question
        └── _export()  →  output/reel_<difficulty>.mp4
```