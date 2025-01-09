import os
import shutil
from tqdm import tqdm
import argparse
from pathlib import Path

from datasets import load_dataset
from collections import defaultdict
import pandas as pd 

def _prepare_cfg():
    parser = argparse.ArgumentParser(description="Segment timit into phone-level audio files.")
    parser.add_argument("--timit_dir", type=Path, help="Base path to the timit data directory")
    parser.add_argument("--phone_dir", type=Path, help="Base path to the phone segment directory")
    parser.add_argument("--dest_dir", type=Path, help="Output path to selected phone segment directory")
    return parser.parse_args()

## This script is used to select unique phone list by each speaker.
def find_unique_phones_by_speaker(timit_dir):
    timit = load_dataset("timit_asr", data_dir=timit_dir)
    unique_phones_by_speaker = {}
    for split in ["train", "test"]:
        for i in tqdm(range(len(timit[split]))):
            fn = timit[split][i]["file"]
            speaker = timit[split][i]["speaker_id"]
            phone_info = timit[split][i]["phonetic_detail"]
            if speaker not in unique_phones_by_speaker:
                unique_phones_by_speaker[speaker] = set()
            unique_phones_by_speaker[speaker].update(phone_info["utterance"])
    return unique_phones_by_speaker

## This script is used to select speakers that includes all phonemes that we want to analyze. 
def select_speakers_with_all_phonemes(unique_phones_by_speaker, analysis):
    selected_speakers = []
    for speaker, phones in unique_phones_by_speaker.items():
        if all(phoneme in phones for phoneme in analysis):
            selected_speakers.append(speaker)
    return selected_speakers

## This script is used to select and copy files that includes the selected speakers and phonemes.
def select_analyze_vowels(ori_path, dest_path, analyze_spks, analyze_phns):
    for r, d, files in os.walk(ori_path):
        for fn in tqdm(files):
            speaker = fn.split("_")[1]
            phn = fn.split(".")[0].split("_")[-1]
            if speaker in analyze_spks:
                if phn in analyze_phns:
                    src = os.path.join(r, fn)
                    speaker_dest_dir = os.path.join(dest_path, speaker)
                    os.makedirs(speaker_dest_dir, exist_ok=True)
                    dest = os.path.join(speaker_dest_dir, fn)
                    shutil.copy(src, dest)

if __name__ == "__main__":
    analysis = "iy ih eh ae ah uw uh ao aa".split()
    args = _prepare_cfg()
    timit_dir = args.timit_dir
    phone_dir = args.phone_dir
    dest_dir = args.dest_dir

    selected_speakers = select_speakers_with_all_phonemes(find_unique_phones_by_speaker(timit_dir), analysis)
    select_analyze_vowels(ori_path=phone_dir, dest_path=dest_dir, analyze_spks=selected_speakers, analyze_phns=analysis)


