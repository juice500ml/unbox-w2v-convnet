import os
import shutil
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
import pandas as pd 

def find_unique_phones_by_speaker(path):
    timit = load_dataset("timit_asr", data_dir=path)
    def collect_phones(split):
        unique_phones_by_speaker = {}
        for i in tqdm(range(len(timit[split]))):
            speaker = timit[split][i]["speaker_id"]
            phone_info = timit[split][i]["phonetic_detail"]
            if speaker not in unique_phones_by_speaker:
                unique_phones_by_speaker[speaker] = set()
            unique_phones_by_speaker[speaker].update(phone_info["utterance"])
        return unique_phones_by_speaker

    train_spk_phoneme = collect_phones("train")
    test_spk_phoneme = collect_phones("test")

    combined_spk_phoneme = train_spk_phoneme.copy()
    for speaker, phones in test_spk_phoneme.items():
        if speaker in combined_spk_phoneme:
            combined_spk_phoneme[speaker].update(phones)
        else:
            combined_spk_phoneme[speaker] = phones
    return combined_spk_phoneme


def select_speakers_with_all_phonemes(combined_spk_phoneme, analysis):
    selected_speakers = []
    for speaker, phones in combined_spk_phoneme.items():
        if all(phoneme in phones for phoneme in analysis):
            selected_speakers.append(speaker)
    return selected_speakers


def select_analyze_vowels(data_path, path, dest, analyze_spks, analyze_phns):
    # speaker_map = map_spkids(data_path)
    os.makedirs(dest, exist_ok=True)
    for r, d, files in os.walk(path):
        for fn in tqdm(files):
            speaker = fn.split("_")[1]
            phn = fn.split(".")[0].split("_")[-1]
            if speaker in analyze_spks:
                if phn in analyze_phns:
                    src_path = os.path.join(r, fn)
                    # nspk = speaker_map[speaker]
                    speaker_dest_dir = os.path.join(dest, speaker)
                    os.makedirs(speaker_dest_dir, exist_ok=True)
                    dest_path = os.path.join(speaker_dest_dir, fn)
                    shutil.copy(src_path, dest_path)

if __name__ == "__main__":
    path = "/data/user_data/eyeo2/data/timit"
    analysis = "iy ih eh ae ah uw uh ao aa".split()
    combined_spk_phoneme = find_unique_phones_by_speaker(path)
    selected_speakers = select_speakers_with_all_phonemes(combined_spk_phoneme, analysis)
    print(len(selected_speakers))
    select_analyze_vowels(data_path=path, path="/data/user_data/eyeo2/data/CP/all_phone_segments/", dest="/data/user_data/eyeo2/data/CP/analyze_vowels2/", analyze_spks=selected_speakers, analyze_phns=analysis)


