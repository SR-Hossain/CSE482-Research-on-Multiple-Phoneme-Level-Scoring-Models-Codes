#!/usr/bin/env python3
"""
run.py ― Dataset recorder for LibriSpeech‑style structure
========================================================
* Expects one or more transcript files named  <speaker_id>.<chapter_id>.trans.txt
  (dot or dash between speaker & chapter also accepted, e.g.  86-123412.trans.txt).
* Creates      <out_dir>/<speaker_id>/<chapter_id>/
* Records FLAC (16 kHz/16‑bit mono) as  <speaker_id>-<chapter_id>-XXXX.flac
* Shows each prompt on‑screen; press Enter to start and stop recording.
* Can be interrupted (Ctrl‑C or [q]) and resumed later; existing audio files
  are detected and skipped so you never re‑record finished lines.

Dependencies
------------
    pip install sounddevice soundfile numpy

Quick start
-----------
    python run.py --speaker_id 86 --transcripts_dir . --out_dir data

    (Or omit --speaker_id to process every transcript in the directory.)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf


import json


# ── constants ──────────────────────────────────────────────────────────
SR = 16_000        # sample‑rate (Hz)
CHANNELS = 1
SUBTYPE = "PCM_16"  # 16‑bit FLAC



SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "user_config.json"
RESOURCES_PATH = SCRIPT_DIR / "resources.json"

def load_or_create_resources() -> dict:
    if RESOURCES_PATH.exists():
        with open(RESOURCES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_resources(resources: dict):
    with open(RESOURCES_PATH, "w", encoding="utf-8") as f:
        json.dump(resources, f, indent=2)

def load_or_create_user_config() -> dict:
    """Load existing user config or prompt the user to create one."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    
    print("🔧 First-time setup: Please enter your user information.\n")
    user_config = {
        "name": input("Your full name: ").strip(),
        "age": input("Your age: ").strip(),
        "gender": input("Your gender (M/F/Other): ").strip(),
        "id": input("Your speaker ID (e.g., 86): ").strip(),
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(user_config, f, indent=2)
    
    print(f"\n✅ User info saved to {CONFIG_PATH.name}")
    return user_config


# ── helpers ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record speech dataset with LibriSpeech naming")
    p.add_argument("--speaker_id", help="Process only this speaker (optional)")
    p.add_argument("--transcripts_dir", default=".", help="Directory containing *.trans.txt files")
    p.add_argument("--out_dir", default=".", help="Root directory where audio folders will be created")
    p.add_argument("--sr", type=int, default=SR, help="Sample rate (default 16 kHz)")
    return p.parse_args()


def discover_transcripts(path: Path) -> List[Tuple[int, Path]]:
    """Find all trans.<index>.txt and return list of (chapter_index, Path)"""
    pattern = re.compile(r"^trans\.(\d+)\.txt$")
    files: List[Tuple[int, Path]] = []
    for f in path.glob("trans.*.txt"):
        m = pattern.match(f.name)
        if not m:
            continue
        chapter_index = int(m.group(1))
        files.append((chapter_index, f))
    if not files:
        sys.exit("No transcript files like trans.<index>.txt found.")
    return sorted(files)




def load_lines(transcript_path: Path) -> List[Tuple[str, str]]:
    """Parse transcript file -> list[(utterance_id, text)]."""
    lines: List[Tuple[str, str]] = []
    with transcript_path.open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            lines.append(raw)

    return lines


def record_once(sr: int) -> np.ndarray:
    """Record until user presses Enter; return numpy array."""
    print("Recording… (press Enter again to stop)")
    chunks: List[np.ndarray] = []
    sd.default.samplerate = sr
    sd.default.channels = CHANNELS
    with sd.InputStream(callback=lambda indata, *_: chunks.append(indata.copy())):
        input()  # wait for Enter
    if not chunks:
        return np.empty((0, CHANNELS))
    return np.concatenate(chunks, axis=0)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ── main ───────────────────────────────────────────────────────────────

def main() -> None:
    user_info = load_or_create_user_config()
    a = parse_args()

    resources = load_or_create_resources()
    transcripts_dir = Path(a.transcripts_dir)
    out_root = Path(a.out_dir)

    if not a.speaker_id:
        a.speaker_id = user_info["id"]

    speaker_id = int(a.speaker_id)
    chapters = discover_transcripts(transcripts_dir)

    for chapter_index, tpath in chapters:
        chapter_id = speaker_id * 1000 + chapter_index

        if str(chapter_id) not in resources:
            print(f"\n📚 Which book or resource was used to create trans.{chapter_index}.txt?")
            resource = input("Enter resource name: ").strip()
            resources[chapter_id] = resource
            save_resources(resources)

        spk_id = str(speaker_id)
        chap_id = str(chapter_id)

        dest_dir = out_root / spk_id / chap_id
        ensure_dir(dest_dir)
        transcript_file = dest_dir / "transcript.txt"

        entries = load_lines(tpath)
        total = len(entries)
        print(f"\n== Speaker {spk_id} | Chapter {chap_id} | {total} lines ==")

        for idx, text in enumerate(entries):
            sentence_index = idx + 1
            utt_id = f"{speaker_id}-{chapter_id:06d}-{sentence_index:04d}"
            wav_path = dest_dir / f"{utt_id}.flac"

            if not wav_path.exists():
                print("\n────────────────────────────────────────")
                print(f"Sentence {idx+1}/{total}: {text}")
                input(">>> Press Enter to start recording <<<")
                break

        try:
            for idx, text in enumerate(entries):
                sentence_index = idx + 1
                utt_id = f"{speaker_id}-{chapter_id:06d}-{sentence_index:04d}"
                wav_path = dest_dir / f"{utt_id}.flac"

                if wav_path.exists():
                    continue  # Already recorded

                next_text = entries[idx+1] if idx+1 < total else None

                audio = record_once(a.sr)
                if audio.size == 0:
                    print("⚠️  No audio captured. Skipping line.")
                    continue

                # Confirmation / redo / quit
                while True:
                    if next_text:
                        print("\n────────────────────────────────────────")
                        print(f'Next Text {idx+2}/{total} === {next_text}')
                    choice = input("[Enter]=keep  [r]=redo  [q]=quit > ").strip().lower() or "k"
                    if choice == "r":
                        print("\n────────────────────────────────────────")
                        print(f"Sentence {idx+1}/{total}: {text}")
                        input(">>> Press Enter to start recording <<<")
                        audio = record_once(a.sr)
                        continue
                    if choice == "q":
                        print("Interrupted by user. All saved recordings remain. Bye!")
                        sys.exit(0)
                    # keep
                    sf.write(wav_path, audio, a.sr, format="FLAC", subtype=SUBTYPE)
                    with transcript_file.open("a", encoding="utf-8") as tf:
                        tf.write(f"{utt_id} {text}\n")
                    break
        except KeyboardInterrupt:
            print("\nInterrupted! Your recordings are safe. Rerun to continue.")
            sys.exit(0)

    print("\n🎉  All transcripts processed. Thank you!")



if __name__ == "__main__":
    main()
