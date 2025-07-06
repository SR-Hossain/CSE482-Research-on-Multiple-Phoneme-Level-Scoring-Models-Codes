import os
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

subset_name = "train-clean-60"
output_file = "CHAPTERS"

def get_duration(file_path):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return 0.0

def get_all_flac_files(folder):
    return [os.path.join(dp, f) for dp, _, files in os.walk(folder) for f in files if f.endswith('.flac')]

def compute_chapter_stats(chapter_path):
    files = get_all_flac_files(chapter_path)
    total = 0.0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(get_duration, f) for f in files]
        for future in as_completed(futures):
            total += future.result()
    return round(total / 60, 2), len(files)  # minutes, file count

def main():
    lines = [";ID        |READER|MINUTES|NUM_FILES| SUBSET         | TITLE"]

    for user_folder in os.listdir("."):
        if not os.path.isdir(user_folder):
            continue

        config_path = os.path.join(user_folder, "user_config.json")
        resources_path = os.path.join(user_folder, "resources.json")

        if not (os.path.isfile(config_path) and os.path.isfile(resources_path)):
            continue

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = json.load(f)
                user_id = user_config.get("id", "UNKNOWN")

            with open(resources_path, "r", encoding="utf-8") as f:
                resources = json.load(f)

            user_audio_root = os.path.join(user_folder, user_id)
            if not os.path.isdir(user_audio_root):
                continue

            for chapter_id in os.listdir(user_audio_root):
                chapter_path = os.path.join(user_audio_root, chapter_id)
                if not os.path.isdir(chapter_path):
                    continue

                minutes, num_files = compute_chapter_stats(chapter_path)
                chapter_title = resources.get(chapter_id, "UNKNOWN")

                line = f"{chapter_id:<10}|{user_id:<6}|{minutes:<7}|{num_files:<9}|{subset_name:<16}|{chapter_title}"
                lines.append(line)

        except Exception as e:
            print(f"Error processing folder {user_folder}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"CHAPTERS file created at: {output_file}")

if __name__ == "__main__":
    main()
