import os
import glob
import parselmouth
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
import argparse
from pathlib import Path
from transformers import AutoModel, Wav2Vec2FeatureExtractor
import torch
import csv

from manipulate_formants import change_formants, copy_mean_intensity

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model directly on GPU
processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
model = AutoModel.from_pretrained("microsoft/wavlm-large").to(device)

# Argument parser setup
def _prepare_cfg():
    parser = argparse.ArgumentParser(description="Interpolate corner vowels.")
    parser.add_argument("--base_path", type=str, default="../corner_vowels/group100", help="Base path to corner vowels directory")
    parser.add_argument("--num_steps", type=int, default=13, help="Number of interpolation steps")
    parser.add_argument("--spk_info", type=Path, default=Path("timit_spk.csv"), help="Speaker info for max formant frequency")
    return parser.parse_args()

# Audio file selection
def select_audio_files(base_path, speaker_id):
    speaker_path = os.path.join(base_path, speaker_id)
    if not os.path.isdir(speaker_path):
        raise ValueError(f"Speaker folder {speaker_path} does not exist.")
    
    sound_files = [f for f in os.listdir(speaker_path) if f.endswith('.wav')]
    valid_pairs = [
        (os.path.join(speaker_path, a), os.path.join(speaker_path, b))
        for a, b in itertools.combinations(sound_files, 2)
        if a.split("_")[-1][:2] != b.split("_")[-1][:2]
    ]
    return valid_pairs

# Midpoint formant extraction
def get_midpoint_formants(sound):
    midpoint = sound.duration / 2
    formant = sound.to_formant_burg()
    return [formant.get_value_at_time(i, midpoint) for i in range(1, 4)]

# Vowel interpolation with matched intensity
def interpolate_vowels(sound, new_f1, new_f2, new_f3, max_formant=5500):
    modified_sound = change_formants(sound, new_f1, new_f2, new_f3, max_formant)
    return copy_mean_intensity(sound, modified_sound)

# Interpolation and feature extraction
def interpolate_sounds(sound_a, sound_b, newfn, num_steps=13, max_formant=5500):
    folder_name = newfn.split("_")[0]
    output_folder = os.path.join(f"interpolated_vowels_{num_steps}", folder_name)
    result_folder = f"result_{num_steps}"
    fig_folder = f"fig_{num_steps}"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(fig_folder, exist_ok=True)
    
    f1_a, f2_a, f3_a = get_midpoint_formants(sound_a)
    f1_b, f2_b, f3_b = get_midpoint_formants(sound_b)
    f1_values_ab = np.linspace(f1_a, f1_b, num_steps)
    f2_values_ab = np.linspace(f2_a, f2_b, num_steps)
    f3_values_ab = np.linspace(f3_a, f3_b, num_steps)

    interpolated_sounds_a, interpolated_sounds_b = [], []
    newfn_b = f"{newfn.split("_")[0]}_{newfn.split("_")[2]}_{newfn.split("_")[1]}"
    for i in range(num_steps):
        output_path_a = os.path.join(output_folder, f"{newfn}_{i}.wav")
        output_path_b = os.path.join(output_folder, f"{newfn_b}_{i}.wav")

        if not os.path.exists(output_path_a):
            interpolated_sound_a = interpolate_vowels(sound_a, f1_values_ab[i], f2_values_ab[i], f3_values_ab[i], max_formant)
            interpolated_sound_a.save(output_path_a, "WAV")
            interpolated_sounds_a.append(output_path_a)
            
            interpolated_sound_b = interpolate_vowels(sound_b, f1_values_ab[num_steps - i - 1], f2_values_ab[num_steps - i - 1], f3_values_ab[num_steps - i - 1], max_formant)
            interpolated_sound_b.save(output_path_b, "WAV")
            interpolated_sounds_b.append(output_path_b)
        
        if os.path.exists(output_path_a):
            interpolated_sounds_a.append(output_path_a)
            interpolated_sounds_b.append(output_path_b)
    
    return interpolated_sounds_a, interpolated_sounds_b

# SSL feature extraction
def extract_ssl_features_from_sound(audio_path, num_steps):
    outputList = []
    for i in range(num_steps):
        sound = parselmouth.Sound(audio_path[i])
        audio = sound.values.flatten()
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=False)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # Extract the mean hidden state and move to CPU
        mean_hidden_state = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        outputList.append(mean_hidden_state)
        # print(mean_hidden_state.shape)
    return outputList

def compute_cosine_similarity(feature_a, feature_b):
    # Ensure the inputs are NumPy arrays
    if isinstance(feature_a, torch.Tensor):
        feature_a = feature_a.cpu().numpy()
    if isinstance(feature_b, torch.Tensor):
        feature_b = feature_b.cpu().numpy()
    
    # Compute cosine similarity
    dot_product = np.dot(feature_a, feature_b)
    norm_a = np.linalg.norm(feature_a)
    norm_b = np.linalg.norm(feature_b)
    cosine_similarity = dot_product / (norm_a * norm_b)
    
    # Ensure output has shape (1,)
    return np.array([cosine_similarity])

    
def normalize_pair(phn_a, phn_b):
    return '_'.join(sorted([phn_a, phn_b]))

# Process speaker pairs
# Process speaker pairs and save similarities
def process_speaker_pairs(base_path, speaker_id, pairs, num_steps=13):
    rows = []
    for sound_a_path, sound_b_path in pairs:
        sound_a, sound_b = parselmouth.Sound(sound_a_path), parselmouth.Sound(sound_b_path)
        phn_a, phn_b = sound_a_path.split("/")[-1].split("_")[1][:2], sound_b_path.split("/")[-1].split("_")[1][:2]
        newfn = f"{speaker_id}_{phn_a}_{phn_b}"
        
        interpolated_sound_a, interpolated_sound_b = interpolate_sounds(sound_a, sound_b, newfn, num_steps)
        featureAList = extract_ssl_features_from_sound(interpolated_sound_a, num_steps)
        featureBList = extract_ssl_features_from_sound(interpolated_sound_b, num_steps)
        
        # Compute cosine similarities for each interpolation step
        similarities_a = [compute_cosine_similarity(featureAList[0], feat)[0] for feat in featureAList]
        similarities_b = [compute_cosine_similarity(featureBList[0], feat)[0] for feat in featureBList]
        
        normalized_phn_pair = normalize_pair(phn_a[:2], phn_b[:2])
        row = [f"{newfn}", normalized_phn_pair] + similarities_a + similarities_b
        rows.append(row)

    columns = ['filename', 'phnA_B'] + [f"A_{i}" for i in range(num_steps)] + [f"B_{i}" for i in range(num_steps)]
    result_file = f'result_{num_steps}/similarities_{speaker_id}.csv'
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(result_file, index=False)
    return result_file


def normalize_list(values):
    min_val = min(values)
    max_val = max(values)
    return [(val - min_val) / (max_val - min_val) for val in values]
    
# Draw plots from cosine similarities
def draw_spk_plot(result_file, speaker_id, num_steps):
    df = pd.read_csv(result_file).drop(columns=['filename'])
    
    for phnA_phnB, group in df.groupby('phnA_B').mean().iterrows():
        phnSplit = phnA_phnB.split("_")
        plt.figure(figsize=(10, 6))
        
        x_values = range(num_steps)
        
        phnAList = normalize_list(list(group[[f"A_{i}" for i in range(num_steps)]]))
        phnBList = normalize_list(list(group[[f"B_{i}" for i in range(num_steps)]])[::-1])
        
        plt.plot(x_values, phnAList, label=f'{phnSplit[0]}', marker='o')
        plt.plot(x_values, phnBList, label=f'{phnSplit[1]}', marker='o')
        
        plt.xlim(0, num_steps -1)
        plt.ylim(0, 1)
        plt.xticks([0, num_steps -1], [phnSplit[0], phnSplit[1]])
        plt.xlabel("Interpolation Step")
        plt.ylabel("Cosine Similarity (Normalized)")
        plt.title(f"Cosine Similarity Interpolation for {speaker_id}: {phnA_phnB}")
        plt.legend()
        
        plt.savefig(f'fig_{num_steps}/{speaker_id}_{phnA_phnB}.pdf', format="pdf")
        plt.close()

def draw_tot_plot(dfPath, num_steps):
    csv_files = glob.glob(os.path.join(dfPath, "*.csv"))
    
    df_list, phnALists, phnBLists = [], [], []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        for phnA_phnB, group in df.groupby('phnA_B'):
            phnSplit = phnA_phnB.split("_")
            phnAList = list(group[[f"A_{i}" for i in range(0, num_steps)]].mean())
            phnBList = list(group[[f"B_{i}" for i in range(0, num_steps)]].mean())[::-1]
        df_list.append(df)
        phnALists.append(normalize_list(phnAList))
        phnBLists.append(normalize_list(phnBList))

    concatenated_df = pd.concat(df_list, ignore_index=True).drop(columns=['filename'])
    for avg_phnA_phnB, group in concatenated_df.groupby('phnA_B').mean().iterrows():
        phnSplit = avg_phnA_phnB.split("_")
        plt.figure(figsize=(10, 6))
        x_values = range(0, num_steps)

        phnAList = normalize_list(list(group[[f"A_{i}" for i in range(0, num_steps)]]))
        phnBList = normalize_list(list(group[[f"B_{i}" for i in range(0, num_steps)]])[::-1])
    
        # for i in range(len(phnALists)):
        #     plt.plot(x_values, phnALists[i], color=(0.4, 0.4, 0.5, 0.5), alpha=0.5)
        #     plt.plot(x_values, phnBLists[i], color=(0.5, 0.4, 0.2, 0.5), alpha=0.5)
        
        plt.plot(x_values, phnAList, label=f'{phnSplit[0]}')
        plt.plot(x_values, phnBList, label=f'{phnSplit[1]}')
        
        # Add circles at each data point
        plt.scatter(x_values, phnAList, color='blue')
        plt.scatter(x_values, phnBList, color='orange')
        
        plt.xlim(0, num_steps -1)
        plt.ylim(0, 1)
        plt.xticks([0, num_steps -1], [phnSplit[0], phnSplit[1]])
        plt.xlabel("Interpolation Step")
        plt.ylabel("Average Cosine Similarity")
        plt.title(f"Average Cosine Similarity for {speaker_id} {avg_phnA_phnB}")
        plt.legend()
        plt.savefig(f'fig_{num_steps}/total_{avg_phnA_phnB}.pdf', format="pdf")
        plt.close()

# Main script
if __name__ == "__main__":
    args = _prepare_cfg()
    base_path, num_steps, spk_info = args.base_path, args.num_steps, args.spk_info
    genderDict = pd.read_csv(spk_info).set_index("ID")["Sex"].to_dict()
    dfList = []
    for speaker_id in tqdm(os.listdir(base_path)):
        if os.path.isdir(os.path.join(base_path, speaker_id)):
            try:
                max_formant = 5000 if genderDict[speaker_id] == "M" else 5500
                valid_pairs = select_audio_files(base_path, speaker_id)
                result_file = process_speaker_pairs(base_path, speaker_id, valid_pairs, num_steps)
                draw_spk_plot(result_file, speaker_id, num_steps)
            except ValueError as e:
                print(f"Error processing {speaker_id}: {e}")
    
    draw_tot_plot(f"result_{num_steps}", num_steps)