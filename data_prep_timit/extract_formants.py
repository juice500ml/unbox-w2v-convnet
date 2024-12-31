import os
import shutil
from tqdm import tqdm
import pandas as pd
from scipy.stats import zscore
import parselmouth
from parselmouth.praat import call
from extract_spk_info_timit import map_sex

def get_formants(sound, max_formant):
    original_duration = sound.get_total_duration()
    formant = call(sound, "To Formant (burg)", 0, 5, max_formant, 0.025, 50)
    f1_mean = call(formant, "Get mean", 1, 0.0, 0.0, "Hertz")
    f2_mean = call(formant, "Get mean", 2, 0.0, 0.0, "Hertz")
    f3_mean = call(formant, "Get mean", 3, 0.0, 0.0, "Hertz")
    return original_duration, f1_mean, f2_mean, f3_mean

def filter_audio_by_formants(root_path, sex_dict):
    if os.path.exists("formants_filtered.csv"):
        print("Reading formants from file...")
        df = pd.read_csv("formants_filtered.csv")
    else:
        print("Calculating formants...")
        audio_data = []
        for r, d, files in tqdm(os.walk(root_path)):
            for file in files:
                if file.endswith(".wav"):
                    spk = file.split("_")[1]
                    sex = sex_dict[spk]
                    if sex == "M":
                        max_formant = 5000
                    elif sex == "F":
                        max_formant = 5500
                    file_path = os.path.join(r, file)
                    sound = parselmouth.Sound(file_path)
                    duration, f1_mean, f2_mean, f3_mean = get_formants(sound, max_formant)
                    audio_data.append({
                        "file": file,
                        "duration": duration,
                        "f1_mean": f1_mean,
                        "f2_mean": f2_mean,
                        "f3_mean": f3_mean,
                    })
        df = pd.DataFrame(audio_data)

        # Calculate z-scores for formants
        df["f1_mean_zscore"] = zscore(df["f1_mean"])
        df["f2_mean_zscore"] = zscore(df["f2_mean"])
        df["f3_mean_zscore"] = zscore(df["f3_mean"])

        # Filter out rows where any formant z-score is an outlier
        df = df[(df["f1_mean_zscore"].abs() <= 3) & 
                        (df["f2_mean_zscore"].abs() <= 3) & 
                        (df["f3_mean_zscore"].abs() <= 3)]

        # Drop z-score columns as they are no longer needed
        df = df.drop(columns=["f1_mean_zscore", "f2_mean_zscore", "f3_mean_zscore"])
        df.to_csv("formants_filtered.csv", index=False)

    file_list = list(df["file"])
    return file_list