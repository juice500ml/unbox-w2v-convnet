import os
import subprocess
from datasets import load_dataset
from pydub import AudioSegment
from tqdm import tqdm
import sys

dest = "/data/user_data/eyeo2/data/CP/all_phone_segments/"

# Function to convert NIST SPHERE to WAV using SoX
def convert_to_wav(sphere_file, wav_file):
    subprocess.run(["sox", sphere_file, wav_file], check=True)

def segment_timit(data_path, output_dir):
    timit = load_dataset("timit_asr", data_dir=data_path)
    for split in ["train", "test"]:
        for i in tqdm(range(len(timit[split]))):
            audio_path = timit[split][i]["file"]
            dr = audio_path.split("/")[-3] ## dialect region
            spk = audio_path.split("/")[-2] ## speaker
            utt = audio_path.split("/")[-1].split(".")[0] ## utterance
            fileinfo = f"{dr}_{spk}_{utt}"
                
            word_info = timit[split][i]["word_detail"]
            phone_info = timit[split][i]["phonetic_detail"]
            audio_file = timit[split][i]["file"]

            word_starts, word_ends, word_utterances = word_info["start"], word_info["stop"], word_info["utterance"]
            phone_starts, phone_ends, phone_utterances = phone_info["start"], phone_info["stop"], phone_info["utterance"]

            # Load the audio file
            convert_to_wav(audio_file, wav_file) ## convert sphere file into wav file 
            wav_file = audio_file.replace(".WAV", ".wav") 
            audio = AudioSegment.from_wav(wav_file)

            current_word = ""
            word_index = 0
            for j in range(len(phone_utterances)):
                phone_start, phone_end, phone_utterance = phone_starts[j], phone_ends[j], phone_utterances[j]
                if phone_utterance == "h#": ## skip silence
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
                filepath = os.path.join(output_dir, split, filename)

                # Export the segment as a .wav file
                segment.export(filepath, format="wav")

if __name__ == "__main__":
    segment_timit(data_path="/data/user_data/eyeo2/data/timit", output_dir="/data/user_data/eyeo2/data/CP/all_phone_segments/")