# DupFrameFixer

Remove consecutive duplicate frames from a video file or an image sequence.

## Features
- Video input: extracts to a temporary TIFF sequence, removes duplicate consecutive frames, renumbers, and re-encodes to a new video with best-effort matching codec/settings.
- Image sequence input: converts to temporary TIFF sequence, removes duplicate consecutive frames, renumbers, and converts back to the original image format into a new output folder.
- Deletes duplicates to the OS trash (Windows/Linux/macOS).
- Similarity threshold adjustable; default is 99.5%.

## Requirements
- Python 3.10+
- ffmpeg + ffprobe available in PATH

## Install ffmpeg
Download and install ffmpeg, then ensure `ffmpeg` and `ffprobe` are available on your PATH:
- https://ffmpeg.org/download.html

## Installation
### Option A: Virtual environment (recommended)
Create and activate a virtual environment, then install dependencies:

Windows (PowerShell):
```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option B: System Python
```
pip install -r requirements.txt
```

## Usage
### Video input
```
python dupframefixer.py path\to\video.mp4
```

### Image sequence input (folder)
```
python dupframefixer.py path\to\sequence_folder
```

### Options
- `--threshold 0-100` (default 99.5)
- `--yes` auto-confirm deletion
- `--keep-tiff` keep temporary TIFF folder

## Output
- Video input: `*_cleaned.<ext>` next to the original file.
- Image sequence input: `<folder>_clean` next to the original folder, with original image format.
- Report CSV saved in the temporary TIFF folder as `duplicate_report.csv`.

## Notes
- Re-encoding uses best-effort settings (codec, pix_fmt, bitrate, profile, level, color tags). Exact bit-perfect matches are not guaranteed for lossy codecs.
- For threshold 100%, exact byte match is used; below 100% uses SSIM (structural similarity).
## License

This project is licensed under the MIT License — see the [MIT License](https://opensource.org/licenses/MIT) for details.
