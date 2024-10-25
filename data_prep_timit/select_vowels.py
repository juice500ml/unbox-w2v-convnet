import os
import shutil
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
import pandas as pd 

def find_unique_phones(path):
    timit = load_dataset("timit_asr", data_dir=path)
    
    # Function to iterate over a dataset split and collect unique phones
    def collect_phones(split):
        phone_counts = defaultdict(int)
        for i in tqdm(range(len(timit[split]))):
            phone_info = list(timit[split][i]["phonetic_detail"]["utterance"])
            for phone in phone_info:
                phone_counts[phone] += 1
        return phone_counts
    
    # Collect phones from both training and test sets
    train_phone_counts = collect_phones("train")
    test_phone_counts = collect_phones("test")
    
    # Combine counts from both splits
    total_phone_counts = defaultdict(int, train_phone_counts)
    for phone, count in test_phone_counts.items():
        total_phone_counts[phone] += count
    
    # Sort phones by name
    sorted_phones = sorted(total_phone_counts.items())
    
    # Print the unique phones and their counts
    print(f"Unique phones ({len(sorted_phones)}):")
    for phone, count in sorted_phones:
        if phone in ['iy', 'ae', 'aa', 'uw', 'ux']:
            print(f"{phone}: {count}")


def find_unique_phones_by_speaker(path):
    timit = load_dataset("timit_asr", data_dir=path)
    unique_phones_by_speaker = {}

    # Function to iterate over a dataset split and collect unique phones by speaker
    def collect_phones(split):
        for i in tqdm(range(len(timit[split]))):
            fn = timit[split][i]["file"]
            parts = fn.split(".")[0].split("/")
            speaker = parts[-2]
            phone_info = timit[split][i]["phonetic_detail"]
            if speaker not in unique_phones_by_speaker:
                unique_phones_by_speaker[speaker] = set()
            unique_phones_by_speaker[speaker].update(phone_info["utterance"])

    # Collect phones from both training and test sets
    collect_phones("train")
    collect_phones("test")

    # Compute the intersection of phones across all speakers
    if unique_phones_by_speaker:
        intersection_phones = set.intersection(*unique_phones_by_speaker.values())
        print(f"Intersection of phones across all speakers ({len(intersection_phones)}): {sorted(intersection_phones)}")
    else:
        print("No phones found.")


# 'iy', 'ae', 'aa', 'uw', 'ux'
# 'ih', 'eh' (alternative exps; iy vs ih or ae vs eh) 
def copy_corner_vowels(path, dest, vowels=['iy', 'ae', 'aa', 'uw', 'ux']):
    os.makedirs(dest, exist_ok=True)
    # Walk through the directory tree
    for r, d, files in os.walk(path):
        for fn in tqdm(files):
            phn = fn.split(".")[0].split("_")[-1]
            if phn in vowels:
                src_path = os.path.join(r, fn)
                dest_path = os.path.join(dest, fn)
                shutil.copy(src_path, dest_path)


def count_corner_vowels(path, vowels=['iy', 'ae', 'aa', 'uw', 'ux']):
    count = {v: 0 for v in vowels}
    count['uw_ux'] = 0  # Combined key for 'uw' and 'ux'
    
    # Walk through the directory tree
    for r, d, files in os.walk(path):
        for fn in tqdm(files):
            phn = fn.split(".")[0].split("_")[-1]
            if phn in vowels:
                if phn in ['uw', 'ux']:
                    count['uw_ux'] += 1
                count[phn] += 1
    print(count)


def count_vowels_by_speaker(path, vowels=['iy', 'ae', 'aa', 'uw', 'ux']):
    # Dictionary to count vowels by speaker
    vowel_counts = defaultdict(lambda: defaultdict(int))
    
    # Walk through the directory tree
    for r, d, files in os.walk(path):
        for fn in tqdm(files):
            parts = fn.split(".")[0].split("_")
            speaker = parts[1]
            phn = parts[-1]
            if phn in vowels:
                if phn in ['uw', 'ux']:
                    vowel_counts[speaker]['uw_ux'] += 1
                vowel_counts[speaker][phn] += 1
    
    # Convert the counts to a DataFrame
    data = []
    for speaker, counts in vowel_counts.items():
        row = {'speaker': speaker}
        for vowel in vowels:
            row[vowel] = counts[vowel]
        row['uw_ux'] = counts['uw_ux']
        data.append(row)
    
    df = pd.DataFrame(data)
    df.fillna(0, inplace=True)  # Fill NaN values with 0
    
    # Write the DataFrame to a CSV file
    df.to_csv('vowel_counts_by_speaker.csv', index=False)


if __name__ == "__main__":
    # find_unique_phones("../../../datasets/timit/TIMIT")
    # find_unique_phones_by_speaker("../../../datasets/timit/TIMIT")
    # copy_corner_vowels("../../../datasets/phone_segments/", dest="../corner_vowels/")
    # count_corner_vowels("../corner_vowels/")
    count_vowels_by_speaker("../corner_vowels/")

