import os
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_duration(file_path):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return 0.0

def get_all_flac_files(root_dir):
    return [os.path.join(dp, f) for dp, _, filenames in os.walk(root_dir) for f in filenames if f.endswith('.flac')]

def compute_user_audio_stats(user_dir, user_id):
    flac_root = os.path.join(user_dir, user_id)
    if not os.path.isdir(flac_root):
        return 0, 0

    flac_files = get_all_flac_files(flac_root)
    total_duration = 0.0

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_file = {executor.submit(get_duration, f): f for f in flac_files}
        for future in as_completed(future_to_file):
            total_duration += future.result()

    total_minutes = total_duration / 60
    return total_minutes, len(flac_files)

def main():
    subset_name = "train-clean-16"
    speakers_file_path = "SPEAKERS"
    lines = [";ID  |SEX| SUBSET          |MINUTES|NUM_FILES| NAME"]

    for user_folder in os.listdir("."):
        if os.path.isdir(user_folder):
            config_path = os.path.join(user_folder, "user_config.json")
            if os.path.isfile(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                        user_id = config.get("id", "UNKNOWN")
                        gender = config.get("gender", "U")
                        name = config.get("name", "UNKNOWN")

                    minutes, num_files = compute_user_audio_stats(user_folder, user_id)
                    line = f"{user_id}|{gender}|{subset_name:<17}|{minutes:<7}|{num_files:<9}|{name}"
                    lines.append(line)

                except Exception as e:
                    print(f"Error processing {user_folder}: {e}")

    with open(speakers_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"SPEAKERS file successfully created at: {speakers_file_path}")

if __name__ == "__main__":
    main()
