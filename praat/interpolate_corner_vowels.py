import os
import parselmouth
import numpy as np
import matplotlib.pyplot as plt
from manipulate_formants import change_formants, copy_mean_intensity
import itertools
from tqdm import tqdm

# Function to extract mean formant frequencies at the midpoint of the sound
def get_midpoint_formants(sound):
    midpoint = sound.duration / 2
    formant = sound.to_formant_burg()
    f1 = formant.get_value_at_time(1, midpoint)
    f2 = formant.get_value_at_time(2, midpoint)
    f3 = formant.get_value_at_time(3, midpoint)
    return f1, f2, f3

# Function to change formants and match intensity
def interpolate_vowels(sound, new_f1, new_f2, new_f3, max_formant=5500):
    fsound = change_formants(sound, new_f1, new_f2, new_f3, max_formant)
    result = copy_mean_intensity(sound, fsound)
    return result

# Function to create interpolated sounds and save them
def generate_interpolated_sounds(sound_a, sound_b, newfn, num_steps=13):
    # Get midpoint formants for both vowels
    f1_a, f2_a, f3_a = get_midpoint_formants(sound_a)
    f1_b, f2_b, f3_b = get_midpoint_formants(sound_b)
    
    # Interpolate formant values between /a/ and /b/
    f1_values = np.linspace(f1_a, f1_b, num_steps)
    f2_values = np.linspace(f2_a, f2_b, num_steps)
    f3_values = np.linspace(f3_a, f3_b, num_steps)
    
    # Generate interpolated sounds
    output_sounds = []
    folder_name = newfn.split("_")[0]
    output_folder = os.path.join("interpolated_vowels", folder_name)
    os.makedirs(output_folder, exist_ok=True)
    
    for i in range(num_steps):
        output_path = os.path.join(output_folder, f"{newfn}_{i}.wav")
        if os.path.exists(output_path):
            print(f"File {output_path} already exists. Skipping.")
            continue
            
        interpolated_sound = interpolate_vowels(sound_a, f1_values[i], f2_values[i], f3_values[i])
        output_sounds.append(interpolated_sound)
        
        # Save each interpolated sound to a file
        output_path = os.path.join(output_folder, f"{newfn}_{i}.wav")
        interpolated_sound.save(output_path, "WAV")
    
    return f1_values, f2_values, f3_values

# Function to draw and save the formant transition plot
def draw_formant_transition(f1_values, f2_values, f3_values, output_path="formant_transition.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(f1_values, marker='o', linestyle='', label="F1")
    plt.plot(f2_values, marker='o', linestyle='', label="F2")
    plt.plot(f3_values, marker='o', linestyle='', label="F3")
    plt.xlabel("Interpolation Step")
    plt.ylabel("Formant Frequency (Hz)")
    plt.title("Formant Transition")
    plt.legend()
    plt.savefig(output_path)

# Function to select audio files based on criteria
def select_audio_files(base_path, speaker_id):
    speaker_path = os.path.join(base_path, speaker_id)
    if not os.path.isdir(speaker_path):
        raise ValueError(f"Speaker folder {speaker_path} does not exist.")
    
    # Collect all sound files in the speaker's folder
    sound_files = [f for f in os.listdir(speaker_path) if f.endswith('.wav')]
    
    # Pair files with different phones
    valid_pairs = []
    for sound_a_file, sound_b_file in itertools.combinations(sound_files, 2):
        phone_a = sound_a_file.split(".")[0].split("_")[-1][0:2]
        phone_b = sound_b_file.split(".")[0].split("_")[-1][0:2]
        
        if phone_a != phone_b:
            sound_a_path = os.path.join(speaker_path, sound_a_file)
            sound_b_path = os.path.join(speaker_path, sound_b_file)
            valid_pairs.append((sound_a_path, sound_b_path))
    
    if not valid_pairs:
        raise ValueError(f"No valid pairs found for speaker {speaker_id}.")
    
    return valid_pairs

# Main script execution
if __name__ == "__main__":
    base_path = "../corner_vowels"
    num_steps = 13

    # Iterate through each speaker's folder
    for speaker_id in tqdm(os.listdir(base_path)):
        print(speaker_id)
        speaker_path = os.path.join(base_path, speaker_id)
        if os.path.isdir(speaker_path):
            try:
                # Select audio files
                valid_pairs = select_audio_files(base_path, speaker_id)
                for sound_a_path, sound_b_path in valid_pairs:
                    sound_a = parselmouth.Sound(sound_a_path)
                    sound_b = parselmouth.Sound(sound_b_path)

                    # Extract base filenames without extension
                    sound_a_part = os.path.splitext(os.path.basename(sound_a_path))[0].split("_")
                    sound_b_part = os.path.splitext(os.path.basename(sound_b_path))[0].split("_")
                    
                    # Process both orders: a_b and b_a
                    newfn_ab = f"{sound_a_part[0]}_{sound_a_part[1]}_{sound_b_part[1]}"
                    f1_values_ab, f2_values_ab, f3_values_ab = generate_interpolated_sounds(sound_a, sound_b, newfn_ab, num_steps)
                    # draw_formant_transition(f1_values_ab, f2_values_ab, f3_values_ab, output_path=f"interpolated_vowels/{newfn_ab}_formant_transition.png")
                    
                    newfn_ba = f"{sound_b_part[0]}_{sound_b_part[1]}_{sound_a_part[1]}"
                    f1_values_ba, f2_values_ba, f3_values_ba = generate_interpolated_sounds(sound_b, sound_a, newfn_ba, num_steps)
                    # draw_formant_transition(f1_values_ba, f2_values_ba, f3_values_ba, output_path=f"interpolated_vowels/{newfn_ba}_formant_transition.png")
            except ValueError as e:
                print(e)