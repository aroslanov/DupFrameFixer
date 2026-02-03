#!/usr/bin/env python3
"""Duplicate frame fixer for videos or image sequences.

Pipeline:
- If input is video: extract TIFF sequence next to the video file.
- If input is image sequence: convert to TIFF temp sequence next to the input folder.
- Detect consecutive duplicate frames by similarity threshold.
- Report and optionally delete duplicates to trash.
- Renumber TIFF sequence to remove gaps.
- If video input: re-encode to same container/codec settings (best effort).
- If image sequence input: convert TIFFs back to original format in a new output folder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from send2trash import send2trash
from skimage.metrics import structural_similarity as ssim
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".mxf",
    ".webm",
    ".mpg",
    ".mpeg",
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".exr",
}

META_FILENAME = "dupframefixer_meta.json"


@dataclass
class VideoInfo:
    codec_name: Optional[str]
    pix_fmt: Optional[str]
    bit_rate: Optional[str]
    profile: Optional[str]
    level: Optional[str]
    avg_frame_rate: Optional[str]
    r_frame_rate: Optional[str]
    color_space: Optional[str]
    color_transfer: Optional[str]
    color_primaries: Optional[str]
    field_order: Optional[str]
    has_audio: bool


@dataclass
class DuplicateMatch:
    index: int
    prev_file: Path
    curr_file: Path
    similarity: float


class DupFrameFixerError(RuntimeError):
    pass


LOGGER = logging.getLogger("dupframefixer")


def run_cmd(cmd: Sequence[str]) -> None:
    LOGGER.debug("Running command: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise DupFrameFixerError(
            f"Command failed: {' '.join(cmd)}\n{result.stderr.strip()}"
        )


def run_ffprobe_json(input_path: Path) -> dict:
    LOGGER.debug("Probing media info: %s", input_path)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(input_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise DupFrameFixerError(
            f"ffprobe failed. Ensure ffprobe is available in PATH.\n{result.stderr.strip()}"
        )
    return json.loads(result.stdout)


def parse_video_info(info: dict) -> VideoInfo:
    streams = info.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    has_audio = any(s for s in streams if s.get("codec_type") == "audio")
    LOGGER.debug("Video stream info: %s", video_stream)
    return VideoInfo(
        codec_name=video_stream.get("codec_name"),
        pix_fmt=video_stream.get("pix_fmt"),
        bit_rate=video_stream.get("bit_rate"),
        profile=video_stream.get("profile"),
        level=str(video_stream.get("level")) if video_stream.get("level") else None,
        avg_frame_rate=video_stream.get("avg_frame_rate"),
        r_frame_rate=video_stream.get("r_frame_rate"),
        color_space=video_stream.get("color_space"),
        color_transfer=video_stream.get("color_transfer"),
        color_primaries=video_stream.get("color_primaries"),
        field_order=video_stream.get("field_order"),
        has_audio=has_audio,
    )


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def natural_sort_key(path: Path) -> List:
    parts = re.split(r"(\d+)", path.name)
    key: List = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def list_images(folder: Path) -> List[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    LOGGER.debug("Found %d image files in %s", len(files), folder)
    return sorted(files, key=natural_sort_key)


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        LOGGER.debug("Removing existing directory: %s", path)
        shutil.rmtree(path)
    LOGGER.debug("Creating directory: %s", path)
    path.mkdir(parents=True, exist_ok=True)


def load_meta(tiff_dir: Path) -> dict:
    meta_path = tiff_dir / META_FILENAME
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("Failed to read metadata: %s", meta_path)
        return {}


def save_meta(tiff_dir: Path, meta: dict) -> None:
    meta_path = tiff_dir / META_FILENAME
    try:
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except OSError:
        LOGGER.warning("Failed to write metadata: %s", meta_path)


def infer_digits_from_images(images: List[Path]) -> int:
    digit_lengths = [len(m[-1]) for m in (re.findall(r"\d+", p.stem) for p in images) if m]
    return max(digit_lengths) if digit_lengths else 6


def infer_digits_from_tiffs(tiff_files: List[Path]) -> int:
    digit_lengths = [len(m[-1]) for m in (re.findall(r"\d+", p.stem) for p in tiff_files) if m]
    return max(digit_lengths) if digit_lengths else 6


def extract_video_to_tiff(video_path: Path, tiff_dir: Path, pix_fmt: Optional[str]) -> None:
    LOGGER.info("Extracting video to TIFF: %s", video_path)
    ensure_empty_dir(tiff_dir)
    output_pattern = str(tiff_dir / "frame_%06d.tiff")
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vsync", "0"]
    if pix_fmt:
        cmd += ["-pix_fmt", pix_fmt]
    cmd += ["-start_number", "1", output_pattern]
    run_cmd(cmd)


def convert_images_to_tiff(images: List[Path], tiff_dir: Path) -> Tuple[str, int]:
    LOGGER.info("Converting image sequence to TIFF: %s", tiff_dir)
    ensure_empty_dir(tiff_dir)
    if not images:
        raise DupFrameFixerError("No images found in the input folder.")
    digit_lengths = [len(m[-1]) for m in (re.findall(r"\d+", p.stem) for p in images) if m]
    max_digits = max(digit_lengths) if digit_lengths else 6
    for idx, img_path in enumerate(images, start=1):
        LOGGER.debug("Converting %s to TIFF (%d/%d)", img_path.name, idx, len(images))
        with Image.open(img_path) as img:
            out_name = f"frame_{idx:0{max_digits}d}.tiff"
            img.save(tiff_dir / out_name, format="TIFF")
    input_ext = images[0].suffix.lower().lstrip(".")
    return input_ext, max_digits


def hash_file(path: Path) -> str:
    LOGGER.debug("Hashing file: %s", path.name)
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_image_array(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        return np.asarray(img)


def compute_similarity(prev_path: Path, curr_path: Path, threshold_percent: float) -> float:
    if threshold_percent >= 100.0:
        LOGGER.debug("Using exact hash match for %s and %s", prev_path.name, curr_path.name)
        return 1.0 if hash_file(prev_path) == hash_file(curr_path) else 0.0
    prev_arr = load_image_array(prev_path)
    curr_arr = load_image_array(curr_path)
    if prev_arr.shape != curr_arr.shape:
        LOGGER.debug("Shape mismatch: %s vs %s", prev_arr.shape, curr_arr.shape)
        return 0.0
    score = ssim(prev_arr, curr_arr, channel_axis=-1)
    return float(score)


def find_duplicates(sequence: List[Path], threshold_percent: float) -> List[DuplicateMatch]:
    duplicates: List[DuplicateMatch] = []
    if len(sequence) < 2:
        return duplicates
    threshold = threshold_percent / 100.0
    LOGGER.info("Scanning %d frames with threshold %.3f", len(sequence), threshold)
    for idx in tqdm(
        range(1, len(sequence)),
        desc="Scanning frames",
        unit="frame",
    ):
        prev_path = sequence[idx - 1]
        curr_path = sequence[idx]
        score = compute_similarity(prev_path, curr_path, threshold_percent)
        LOGGER.debug("Similarity %s -> %s: %.6f", prev_path.name, curr_path.name, score)
        if score >= threshold:
            duplicates.append(
                DuplicateMatch(index=idx, prev_file=prev_path, curr_file=curr_path, similarity=score)
            )
    return duplicates


def write_report(report_path: Path, matches: List[DuplicateMatch]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing report: %s", report_path)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "prev_file", "curr_file", "similarity"])
        for match in matches:
            writer.writerow([
                match.index,
                match.prev_file.name,
                match.curr_file.name,
                f"{match.similarity:.6f}",
            ])


def prompt_confirm() -> bool:
    response = input("Delete consecutive duplicates to trash? [y/N]: ").strip().lower()
    return response == "y" or response == "yes"


def delete_to_trash(files: Iterable[Path]) -> None:
    for file_path in files:
        LOGGER.debug("Sending to trash: %s", file_path)
        send2trash(str(file_path))


def renumber_sequence(tiff_dir: Path) -> None:
    LOGGER.info("Renumbering TIFF sequence: %s", tiff_dir)
    files = sorted(tiff_dir.glob("*.tiff"), key=natural_sort_key)
    if not files:
        return
    digit_lengths = [len(m[-1]) for m in (re.findall(r"\d+", f.stem) for f in files) if m]
    digits = max(digit_lengths) if digit_lengths else 6
    temp_files = []
    for idx, file_path in enumerate(files, start=1):
        temp_name = f"__tmp__{idx:0{digits}d}.tiff"
        temp_path = tiff_dir / temp_name
        LOGGER.debug("Temp rename %s -> %s", file_path.name, temp_name)
        file_path.rename(temp_path)
        temp_files.append(temp_path)
    for idx, temp_path in enumerate(temp_files, start=1):
        final_name = f"frame_{idx:0{digits}d}.tiff"
        LOGGER.debug("Final rename %s -> %s", temp_path.name, final_name)
        temp_path.rename(tiff_dir / final_name)


def convert_tiff_to_output(
    tiff_dir: Path,
    output_dir: Path,
    output_ext: str,
    prefix: str,
    digits: int,
) -> None:
    LOGGER.info("Converting TIFF sequence back to %s: %s", output_ext, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(tiff_dir.glob("*.tiff"), key=natural_sort_key)
    for idx, tiff_path in enumerate(files, start=1):
        out_name = f"{prefix}{idx:0{digits}d}.{output_ext}"
        LOGGER.debug("Saving %s (%d/%d)", out_name, idx, len(files))
        with Image.open(tiff_path) as img:
            img.save(output_dir / out_name)


def extract_audio_if_present(video_path: Path, temp_dir: Path, has_audio: bool) -> Optional[Path]:
    if not has_audio:
        return None
    LOGGER.info("Extracting audio from video: %s", video_path)
    audio_path = temp_dir / "audio.mka"
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-c:a", "copy", str(audio_path)]
    run_cmd(cmd)
    return audio_path if audio_path.exists() else None


def parse_frame_rate(rate_str: Optional[str]) -> Optional[str]:
    if not rate_str or rate_str == "0/0":
        return None
    return rate_str


def _sanitize_level(level: Optional[str]) -> Optional[str]:
    if not level:
        return None
    try:
        level_int = int(level)
    except (TypeError, ValueError):
        return None
    return str(level_int) if level_int > 0 else None


def _sanitize_bit_rate(codec_name: Optional[str], bit_rate: Optional[str]) -> Optional[str]:
    if not bit_rate:
        return None
    if codec_name and "prores" in codec_name.lower():
        return None
    try:
        br = int(bit_rate)
    except (TypeError, ValueError):
        return None
    return str(br) if br > 0 else None


def _sanitize_pix_fmt(codec_name: Optional[str], pix_fmt: Optional[str]) -> Optional[str]:
    if not pix_fmt:
        return None
    if codec_name and "prores" in codec_name.lower():
        allowed = {"yuv422p10le", "yuv444p10le", "yuva444p10le"}
        return pix_fmt if pix_fmt in allowed else None
    return pix_fmt


def _sanitize_profile(codec_name: Optional[str], profile: Optional[str]) -> Optional[str]:
    if not profile or not codec_name:
        return None
    codec = codec_name.lower()
    prof = profile.strip().lower().replace(" ", "").replace("-", "").replace("_", "")

    if "prores" in codec:
        mapping = {
            "apco": "0",
            "proxy": "0",
            "0": "0",
            "apcs": "1",
            "lt": "1",
            "1": "1",
            "apcn": "2",
            "standard": "2",
            "422": "2",
            "2": "2",
            "apch": "3",
            "hq": "3",
            "422hq": "3",
            "3": "3",
            "ap4h": "4",
            "4444": "4",
            "4": "4",
            "ap4x": "5",
            "4444xq": "5",
            "xq": "5",
            "5": "5",
        }
        return mapping.get(prof)

    if codec in {"h264", "libx264"}:
        mapping = {
            "baseline": "baseline",
            "main": "main",
            "high": "high",
            "high10": "high10",
            "high422": "high422",
            "high444": "high444",
            "constrainedbaseline": "constrained_baseline",
            "constrainedhigh": "constrained_high",
        }
        return mapping.get(prof)

    if codec in {"hevc", "h265", "libx265"}:
        mapping = {
            "main": "main",
            "main10": "main10",
            "mainstillpicture": "mainstillpicture",
        }
        return mapping.get(prof)

    return None


def reencode_video_from_tiff(
    tiff_dir: Path,
    output_video: Path,
    info: VideoInfo,
    audio_path: Optional[Path],
) -> None:
    LOGGER.info("Re-encoding video to: %s", output_video)
    input_pattern = str(tiff_dir / "frame_%06d.tiff")
    frame_rate = parse_frame_rate(info.avg_frame_rate) or parse_frame_rate(info.r_frame_rate)
    cmd = ["ffmpeg", "-y"]
    if frame_rate:
        cmd += ["-framerate", frame_rate]
    cmd += ["-start_number", "1", "-i", input_pattern]
    if audio_path:
        cmd += ["-i", str(audio_path)]
    if info.codec_name:
        cmd += ["-c:v", info.codec_name]

    pix_fmt = _sanitize_pix_fmt(info.codec_name, info.pix_fmt)
    if pix_fmt:
        cmd += ["-pix_fmt", pix_fmt]

    bit_rate = _sanitize_bit_rate(info.codec_name, info.bit_rate)
    if bit_rate:
        cmd += ["-b:v", bit_rate]

    profile = _sanitize_profile(info.codec_name, info.profile)
    if profile:
        cmd += ["-profile:v", profile]

    level = _sanitize_level(info.level)
    if level:
        cmd += ["-level", level]
    if info.color_space:
        cmd += ["-colorspace", info.color_space]
    if info.color_transfer:
        cmd += ["-color_trc", info.color_transfer]
    if info.color_primaries:
        cmd += ["-color_primaries", info.color_primaries]
    if info.field_order:
        cmd += ["-field_order", info.field_order]
    if audio_path:
        cmd += ["-c:a", "copy", "-shortest"]
    cmd += [str(output_video)]
    run_cmd(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove consecutive duplicate frames from a video or image sequence.")
    parser.add_argument("input", help="Input video file or image sequence folder")
    parser.add_argument("--threshold", type=float, default=99.5, help="Similarity threshold percent (0-100). Default: 99.5")
    parser.add_argument("--yes", action="store_true", help="Delete duplicates without prompting")
    parser.add_argument("--remove-temp", action="store_true", help="Delete temporary TIFF folder after processing")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    try:
        args = parser.parse_args()
    except SystemExit as e:
        # If user asked for help, allow normal exit
        if getattr(e, "code", None) == 0:
            raise
        # For missing/invalid args, print a friendly message and exit gracefully
        print("Error: Missing or invalid arguments. Use --help for usage.")
        return 2

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )

    LOGGER.debug("Arguments: %s", args)

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise DupFrameFixerError("Input path does not exist.")

    is_video = is_video_file(input_path)
    if not is_video and not input_path.is_dir():
        raise DupFrameFixerError("Input must be a video file or an image sequence folder.")

    if args.threshold < 0 or args.threshold > 100:
        raise DupFrameFixerError("Threshold must be between 0 and 100.")

    LOGGER.info("Input: %s", input_path)
    LOGGER.info("Mode: %s", "video" if is_video else "image-sequence")
    LOGGER.info("Threshold: %.2f", args.threshold)

    temp_parent = input_path.parent if is_video else input_path.parent
    temp_name = f"{input_path.stem}_tiff_tmp" if is_video else f"{input_path.name}_tiff_tmp"
    tiff_dir = temp_parent / temp_name
    LOGGER.info("Temporary TIFF folder: %s", tiff_dir)

    resume_existing = False
    meta = {}
    if tiff_dir.exists():
        existing_tiffs = sorted(tiff_dir.glob("*.tiff"), key=natural_sort_key)
        if existing_tiffs:
            resume_existing = True
            meta = load_meta(tiff_dir)
            LOGGER.info("Resuming from existing TIFF folder (preserving intermediates).")
        else:
            ensure_empty_dir(tiff_dir)

    input_ext = None
    digits = 6
    prefix = "frame_"
    video_info = None
    audio_path = None

    if is_video:
        info_json = run_ffprobe_json(input_path)
        video_info = parse_video_info(info_json)
        if not resume_existing:
            extract_video_to_tiff(input_path, tiff_dir, video_info.pix_fmt)
        audio_path = None
        audio_candidate = tiff_dir / "audio.mka"
        if audio_candidate.exists():
            audio_path = audio_candidate
        if not audio_path:
            audio_path = extract_audio_if_present(input_path, tiff_dir, video_info.has_audio)
        save_meta(tiff_dir, {
            "is_video": True,
            "audio_path": str(audio_path) if audio_path else None,
        })
    else:
        images = list_images(input_path)
        if not images:
            raise DupFrameFixerError("No images found in the input folder.")
        if resume_existing:
            input_ext = meta.get("input_ext") or images[0].suffix.lower().lstrip(".")
            digits = int(meta.get("digits") or infer_digits_from_images(images))
            prefix = meta.get("prefix") or infer_prefix(images[0].name)
        else:
            input_ext, digits = convert_images_to_tiff(images, tiff_dir)
            prefix = infer_prefix(images[0].name)
        LOGGER.debug("Inferred output prefix: %s", prefix)
        save_meta(tiff_dir, {
            "is_video": False,
            "input_ext": input_ext,
            "digits": digits,
            "prefix": prefix,
        })

    tiff_files = sorted(tiff_dir.glob("*.tiff"), key=natural_sort_key)
    LOGGER.debug("TIFF frames loaded: %d", len(tiff_files))
    matches = find_duplicates(tiff_files, args.threshold)

    report_path = tiff_dir / "duplicate_report.csv"
    write_report(report_path, matches)

    if matches:
        print(f"Found {len(matches)} duplicate consecutive frames.")
        print(f"Report: {report_path}")
        if args.yes or prompt_confirm():
            LOGGER.info("Deleting %d duplicates to trash.", len(matches))
            delete_to_trash(m.curr_file for m in matches)
            renumber_sequence(tiff_dir)
        else:
            LOGGER.info("User declined deletion.")
            print("No files were deleted.")
    else:
        print("No consecutive duplicates detected.")

    if matches:
        if is_video:
            if not video_info:
                raise DupFrameFixerError("Missing video info for re-encode.")
            output_video = input_path.with_name(f"{input_path.stem}_cleaned{input_path.suffix}")
            reencode_video_from_tiff(tiff_dir, output_video, video_info, audio_path)
            print(f"Output video: {output_video}")
        else:
            if not input_ext:
                raise DupFrameFixerError("Missing input image format.")
            output_dir = input_path.parent / f"{input_path.name}_clean"
            convert_tiff_to_output(tiff_dir, output_dir, input_ext, prefix, digits)
            print(f"Output sequence: {output_dir}")
    else:
        LOGGER.info("No duplicates found; skipping re-export.")

    if args.remove_temp:
        LOGGER.info("Removing temporary TIFF folder: %s", tiff_dir)
        shutil.rmtree(tiff_dir, ignore_errors=True)

    return 0


def infer_prefix(filename: str) -> str:
    name = Path(filename).stem
    match = re.search(r"^(.*?)(\d+)$", name)
    if match:
        return match.group(1)
    return "frame_"


if __name__ == "__main__":
    try:
        rc = main()
    except DupFrameFixerError as exc:
        print(f"Error: {exc}")
        rc = 1
    except Exception as exc:  # pragma: no cover - unexpected
        LOGGER.exception("Unhandled exception")
        print(f"Error: {exc}")
        rc = 1
    # Use os._exit to terminate without a visible SystemExit exception in some debuggers
    import os
    os._exit(rc if isinstance(rc, int) else 0)
