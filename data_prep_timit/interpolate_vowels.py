import os
import numpy as np
from pathlib import Path
import itertools
from tqdm import tqdm
import argparse

import parselmouth
from parselmouth.praat import call

from manipulate_formants import change_formants
from extract_spk_info_timit import map_sex
from extract_formants import filter_audio_by_formants

# Function to prepare the argument parser
def _prepare_cfg():
    parser = argparse.ArgumentParser(description="Interpolate vowels.")
    parser.add_argument("--base_path", type=str, help="Base path to the vowels directory")
    parser.add_argument("--num_interpolation", type=int, default=13, help="Number of steps for interpolation")
    parser.add_argument("--spk_info", type=Path, default=Path("data_prep_timit/timit_spk.csv"), help="Speaker info for maximum formant frequency (5000 for male, 5500 for female)")
    parser.add_argument("--output_dir", type=Path, help="Output directory for the interpolated sounds")
    return parser.parse_args()

def group_files_by_phone(speaker_path, file_list):
    """Group files by their vowel phones for phoneme index."""
    sound_files = [f for f in os.listdir(speaker_path) if f.endswith('.wav') and f in file_list]
    grouped_files = {}
    for file in sound_files:
        phone = file.split(".")[0].split("_")[-1]
        if phone not in grouped_files:
            grouped_files[phone] = []
        grouped_files[phone].append(os.path.join(speaker_path, file))
    return grouped_files

def get_avg_formants(sound, max_formant):
    """Get average formants F1, F2, F3 from the sound."""
    formant = call(sound, "To Formant (burg)", 0, 5, max_formant, 0.025, 50)
    f1_mean = call(formant, "Get mean", 1, 0.0, 0.0, "Hertz")
    f2_mean = call(formant, "Get mean", 2, 0.0, 0.0, "Hertz")
    f3_mean = call(formant, "Get mean", 3, 0.0, 0.0, "Hertz")
    return f1_mean, f2_mean, f3_mean

def manipulate_vowels(sound, new_f1, new_f2, new_f3, max_formant):
    result = change_formants(sound, new_f1, new_f2, new_f3, max_formant)
    return result

def save_interpolated_sounds_by_speaker(grouped_files, max_formant, num_steps, output_dir, speaker_id):
    """
    Create interpolated sounds between all combinations of grouped vowel files for a single speaker.
    """
    # Create output folder for speaker
    output_folder = os.path.join(output_dir, speaker_id)
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all pairs of phones
    for (phone_a, files_a), (phone_b, files_b) in itertools.combinations(grouped_files.items(), 2):
        # Iterate through all combinations of files within each phone group
        for i, file_a in enumerate(files_a, start=1):
            for j, file_b in enumerate(files_b, start=1):
                # Load sounds
                sound_a = parselmouth.Sound(file_a)
                sound_b = parselmouth.Sound(file_b)

                # Get average formants
                f1_a, f2_a, f3_a = get_avg_formants(sound_a, max_formant)
                f1_b, f2_b, f3_b = get_avg_formants(sound_b, max_formant)

                # Interpolate formants
                f1_values = np.linspace(f1_a, f1_b, num_steps)
                f2_values = np.linspace(f2_a, f2_b, num_steps)
                f3_values = np.linspace(f3_a, f3_b, num_steps)

                # Generate interpolated sounds for each step
                for step in range(num_steps):
                    # Create filename for a->b
                    newfn_ab = f"{speaker_id}_{phone_a}{i}_{phone_b}{j}_step{step}"
                    output_path_ab = os.path.join(output_folder, f"{newfn_ab}.wav")
                    if not os.path.exists(output_path_ab):
                        interpolated_sound_ab = manipulate_vowels(
                            sound_a, f1_values[step], f2_values[step], f3_values[step], max_formant
                        )
                        interpolated_sound_ab.save(output_path_ab, "WAV")

                    # Create filename for b->a
                    newfn_ba = f"{speaker_id}_{phone_b}{j}_{phone_a}{i}_step{num_steps - step - 1}"
                    output_path_ba = os.path.join(output_folder, f"{newfn_ba}.wav")
                    if not os.path.exists(output_path_ba):
                        interpolated_sound_ba = manipulate_vowels(
                            sound_b, f1_values[step], f2_values[step], f3_values[step], max_formant
                        )
                        interpolated_sound_ba.save(output_path_ba, "WAV")

if __name__ == "__main__":
    args = _prepare_cfg()
    base_path = args.base_path
    output_path = args.output_dir
    num_steps = args.num_interpolation
    spk_info = args.spk_info

    sexDict = map_sex(spk_info)
    file_list = filter_audio_by_formants(base_path, sexDict)

    for speaker_id in tqdm(os.listdir(base_path)):
        speaker_path = os.path.join(base_path, speaker_id)
        if os.path.isdir(speaker_path):
            max_formant = 5000 if sexDict[speaker_id] == "M" else 5500
            grouped_files = group_files_by_phone(speaker_path, file_list)
            save_interpolated_sounds_by_speaker(grouped_files, max_formant, num_steps, output_path, speaker_id)
