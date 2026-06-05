import json
import os
import argparse
from typing import Dict, List, Optional

import pandas as pd
import pysrt
from tqdm import tqdm

BASE_DIR = "data/EgoLife"

# If the caption is invalid, it will automatically enter transcript-only mode
DENSE_CAPTION_DIR = ""
TRANSLATED_DIR = ""

TRANSCRIPT_DIR = os.path.join(BASE_DIR, "EgoLifeCap/Transcript")
SYNC_DIR = os.path.join(BASE_DIR, "EgoLifeCap/Sync")

# In transcript-only mode, only process this person
TARGET_NAME = "A1_JAKE"


def is_valid_dir(path: str) -> bool:
    return bool(path) and os.path.isdir(path)


def get_available_transcript_files(target_name: Optional[str] = None) -> List[Dict[str, str]]:
    transcript_files = []
    if not is_valid_dir(TRANSCRIPT_DIR):
        return transcript_files

    for name in sorted(os.listdir(TRANSCRIPT_DIR)):
        if target_name is not None and name != target_name:
            continue

        name_dir = os.path.join(TRANSCRIPT_DIR, name)
        if not os.path.isdir(name_dir):
            continue

        for date in sorted(os.listdir(name_dir)):
            date_dir = os.path.join(name_dir, date)
            if not os.path.isdir(date_dir):
                continue

            for file in sorted(os.listdir(date_dir)):
                if file.endswith(".srt"):
                    transcript_files.append(
                        {
                            "name": name,
                            "date": date,
                            "file_name": file[:-4],  # 去掉 .srt
                            "transcript_path": os.path.join(date_dir, file),
                        }
                    )
    return transcript_files


def get_captions(caption_path: str) -> List[Dict[str, str]]:
    captions = []
    if not os.path.exists(caption_path):
        return captions

    with open(caption_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # Skip bad lines to prevent the entire sync from crashing
                continue

            custom_id = data.get("custom_id", "")
            if not custom_id:
                continue

            parts = custom_id.split("-")
            if len(parts) < 2:
                continue

            try:
                start = int(parts[-2])
                end = int(parts[-1])
            except ValueError:
                continue

            text = (data.get("translated_text") or "").strip()
            captions.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "type": "caption",
                }
            )
    return captions


def get_transcripts(transcript_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(transcript_path):
        return []

    subs = list(pysrt.open(transcript_path))
    base_name = os.path.splitext(os.path.basename(transcript_path))[0]

    # Get hour from the last part of the file name, e.g., xxx_100000 -> 10
    last_part = base_name.split("_")[-1]
    if len(last_part) < 2 or not last_part[:2].isdigit():
        return []

    hour = int(last_part[:2])
    transcripts: List[Dict[str, str]] = []

    for sub in subs:
        lines = sub.text.split("\n")
        text = lines[-1].strip() if lines else ""

        elem = {
            "start": int(f"{(hour + sub.start.hours):02d}{sub.start.minutes:02d}{sub.start.seconds:02d}"),
            "end": int(f"{(hour + sub.end.hours):02d}{sub.end.minutes:02d}{sub.end.seconds:02d}"),
            "text": text,
            "type": "transcript",
        }
        transcripts.append(elem)

    return transcripts


def find_video(files: List[str], start_time: int, end_time: int):
    for file in files:
        try:
            video_time = int(os.path.splitext(file)[0].split("_")[-1])
        except (ValueError, IndexError):
            continue

        if start_time <= video_time < end_time:
            return file
    return None


def handle_time(time: int):
    if time % 10000 == 6000:
        time = time + 4000
    if time % 1000000 == 600000:
        time = time + 400000
    return time


def parse_time_from_file_name(file_name: str) -> Optional[int]:
    """
    Parse time from the end of the file name.
    For example:
    A1_JAKE_20240223_100000 -> 100000
    A1_JAKE_20240223_10000000.mp4 -> 10000000
    """
    base_name = os.path.splitext(file_name)[0]
    last_part = base_name.split("_")[-1]

    if not last_part.isdigit():
        return None

    return int(last_part)


def match_with_video(info: tuple, all_df: pd.DataFrame):
    name, date, time = info

    if isinstance(time, str):
        parsed_time = parse_time_from_file_name(time)
        if parsed_time is None:
            print(f"Skip invalid time: {time}")
            return []
        time = parsed_time

    root_dir = os.path.join(BASE_DIR, name, date)

    if not os.path.isdir(root_dir):
        return []

    all_video_files = sorted(
        [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
    )

    start_time = time
    end_time = start_time + 3000

    results = []
    while start_time < time + 1000000:
        video_file = find_video(all_video_files, start_time, end_time)
        if video_file is not None:
            cur_df = all_df[
                (all_df["start"] >= (start_time // 100)) &
                (all_df["end"] <= (end_time // 100))
            ]

            if len(cur_df) > 0:
                data = []
                for _, row in cur_df.iterrows():
                    data.append(
                        {
                            "start": row["start"],
                            "end": row["end"],
                            "text": row["text"],
                            "type": row["type"],
                        }
                    )

                results.append({"video_file": video_file, "data": data})

        start_time = end_time
        end_time = handle_time(end_time + 3000)

    return results


def main():
    global BASE_DIR, DENSE_CAPTION_DIR, TRANSLATED_DIR, TRANSCRIPT_DIR, SYNC_DIR, TARGET_NAME

    parser = argparse.ArgumentParser(description="Generate EgoLife caption/transcript sync files.")
    parser.add_argument("--person", default=TARGET_NAME, help="Subject ID used in transcript-only mode.")
    parser.add_argument("--base-dir", default=BASE_DIR, help="EgoLife data root.")
    parser.add_argument("--dense-caption-dir", default=DENSE_CAPTION_DIR)
    parser.add_argument("--translated-dir", default=TRANSLATED_DIR)
    parser.add_argument("--transcript-dir", default=None)
    parser.add_argument("--sync-dir", default=None)
    args = parser.parse_args()

    BASE_DIR = args.base_dir
    DENSE_CAPTION_DIR = args.dense_caption_dir
    TRANSLATED_DIR = args.translated_dir
    TRANSCRIPT_DIR = args.transcript_dir or os.path.join(BASE_DIR, "EgoLifeCap/Transcript")
    SYNC_DIR = args.sync_dir or os.path.join(BASE_DIR, "EgoLifeCap/Sync")
    TARGET_NAME = args.person

    os.makedirs(SYNC_DIR, exist_ok=True)

    caption_mode = False
    translated_files = []

    # Check if the caption directory is valid and contains jsonl files
    if is_valid_dir(DENSE_CAPTION_DIR) and is_valid_dir(TRANSLATED_DIR):
        translated_files = sorted(
            [f for f in os.listdir(TRANSLATED_DIR) if f.endswith(".jsonl")]
        )
        if translated_files:
            caption_mode = True

    if caption_mode:
        print("If DenseCaption / translated is valid, perform sync for both captions and transcripts.")
        iterable = tqdm(translated_files, desc="Syncing captions + transcripts")

        for caption_file in iterable:
            file_name = caption_file[:-6]  # 去掉 .jsonl
            parts = file_name.split("_")
            if len(parts) != 4:
                print(f"Skip invalid caption file name: {file_name}")
                continue

            idx, name, date, time = parts
            name = idx + "_" + name

            caption_path = os.path.join(TRANSLATED_DIR, caption_file)
            transcript_path = os.path.join(TRANSCRIPT_DIR, name, date, f"{file_name}.srt")

            captions = get_captions(caption_path)
            transcripts = get_transcripts(transcript_path)

            all_data = captions + transcripts
            if not all_data:
                results = []
            else:
                all_df = pd.DataFrame(all_data)
                all_df.sort_values(by="start", inplace=True)
                results = match_with_video((name, date, time), all_df)

            with open(os.path.join(SYNC_DIR, f"{file_name}.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    else:
        print(f"If DenseCaption / translated is empty or invalid, perform sync for transcripts only (only {TARGET_NAME}).")
        transcript_items = get_available_transcript_files(target_name=TARGET_NAME)

        for item in tqdm(transcript_items, desc=f"Syncing transcripts only: {TARGET_NAME}"):
            name = item["name"]
            date = item["date"]
            file_name = item["file_name"]
            transcript_path = item["transcript_path"]

            time = parse_time_from_file_name(file_name)
            if time is None:
                print(f"Skip invalid transcript file name: {file_name}")
                continue

            transcripts = get_transcripts(transcript_path)
            if not transcripts:
                results = []
            else:
                all_df = pd.DataFrame(transcripts)
                all_df.sort_values(by="start", inplace=True)
                results = match_with_video((name, date, time), all_df)

            with open(os.path.join(SYNC_DIR, f"{file_name}.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
