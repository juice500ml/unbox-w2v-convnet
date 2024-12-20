import os
import subprocess
from datasets import load_dataset
from pydub import AudioSegment
from tqdm import tqdm
import sys

version = sys.argv[1] # "train" or "test"
dest = "/data/user_data/eyeo2/data/CP/all_phone_segments/"

# Function to convert NIST SPHERE to WAV using SoX
def convert_to_wav(sphere_file, wav_file):
    subprocess.run(["sox", sphere_file, wav_file], check=True)

# Load the TIMIT dataset
timit = load_dataset("timit_asr", data_dir="/data/user_data/eyeo2/data/timit")
output_dir = dest + version
os.makedirs(output_dir, exist_ok=True)

path = "/data/user_data/eyeo2/data/timit"
for i in tqdm(range(len(timit[version]))):
    audio_path = timit[version][i]["file"]
    dr = audio_path.split("/")[-3]
    spk = audio_path.split("/")[-2]
    utt = audio_path.split("/")[-1].split(".")[0]

    fileinfo = f"{dr}_{spk}_{utt}"
        
    word_info = timit[version][i]["word_detail"]
    phone_info = timit[version][i]["phonetic_detail"]
    audio_file = timit[version][i]["file"]

    wav_file = audio_file.replace(".WAV", ".wav")
    convert_to_wav(audio_file, wav_file)

    word_starts, word_ends, word_utterances = word_info["start"], word_info["stop"], word_info["utterance"]
    phone_starts, phone_ends, phone_utterances = phone_info["start"], phone_info["stop"], phone_info["utterance"]

    # Load the audio file
    audio = AudioSegment.from_wav(wav_file)

    current_word = "test"
    word_index = 0

    # Iterate over the phonetic details
    for j in range(len(phone_utterances)):
        phone_start, phone_end, phone_utterance = phone_starts[j], phone_ends[j], phone_utterances[j]

        if phone_utterance == "h#":
            continue
        else:
            if word_index < len(word_starts) and phone_start == word_starts[word_index]:
                current_word = word_utterances[word_index]
                word_index += 1

        # Convert start and end times from sample indices to milliseconds
        start_ms = int(phone_start / 16)  # 16 kHz sample rate
        end_ms = int(phone_end / 16)

        # Segment the audio
        segment = audio[start_ms:end_ms]

        # Create the filename
        filename = f"{fileinfo}_{j}_{current_word}_{phone_utterance}.wav"
        filepath = os.path.join(output_dir, filename)

        # Export the segment as a .wav file
        segment.export(filepath, format="wav")
