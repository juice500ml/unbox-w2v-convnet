import glob
from pathlib import Path
from tqdm import tqdm
import argparse
import pandas as pd
import parselmouth
import os

def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, help="Path to dataset")
    parser.add_argument("--dataset_type", type=str, choices=["timit_authentic", "timit_synthetic"])
    parser.add_argument("--num_interpolation", type=int, default=None)
    parser.add_argument("--output_path", type=Path, help="Output csv folder")
    return parser.parse_args()

def map_splits(df_path="spk_info.csv"):
    df = pd.read_csv(df_path)
    spk_split_map = dict(zip(list(df['speaker']), list(df['split'])))
    return spk_split_map

## audio_path, filename, speaker, phoneme, duration, split
def _prepare_authentic_timit(authentic_timit_path: Path, df_path="spk_info.csv"):
    spk_split_map = map_splits(df_path)
    rows = []
    for p in tqdm(authentic_timit_path.glob("**/*.wav")):
        audio_path = str(p)
        filename = p.stem
        split = p.parent.name
        spk = p.stem.split("_")[1] 
        split = spk_split_map[spk]
        phn = filename.split("_")[-1]
        duration = parselmouth.Sound(audio_path).get_total_duration()
        if duration < 0.025:
            continue
        rows.append(
            {
                "audio_path": audio_path,
                "filename": filename,
                "speaker": spk,
                "phonemes": phn,
                "duration": duration,
                "split": split
            }
        )
    return pd.DataFrame(rows)

## audio_path, filename, speaker, phoneme, duration
def _prepare_synthetic_timit(synthetic_timit_path: Path):
    rows = []
    for p in tqdm(synthetic_timit_path.glob("**/*.wav")):
        audio_path = str(p)
        filename = p.stem
        spk = p.stem.split("_")[0] 
        phn = filename.split("_")[1] + "_" + filename.split("_")[2]
        duration = parselmouth.Sound(audio_path).get_total_duration()
        if duration < 0.025:
            continue
        rows.append(
            {
                "audio_path": audio_path,
                "filename": filename,
                "speaker": spk,
                "phonemes": phn,
                "iteration": filename.split(".")[0].split("_")[-1],
                "duration": duration,
                "split": "test"
            }
        )
    return pd.DataFrame(rows)

if __name__ == "__main__":
    args = _get_args()
    _prepare = {
        "timit_authentic": _prepare_authentic_timit,
        "timit_synthetic": _prepare_synthetic_timit,
    }[args.dataset_type]
    df = _prepare(args.dataset_path)

    os.makedirs(args.output_path, exist_ok=True)
    if args.num_interpolation is not None:
        csv_path = args.output_path / f"{args.dataset_type}_{args.num_interpolation}.original.pkl"
    else:
        csv_path = args.output_path / f"{args.dataset_type}.original.pkl"
    df.to_pickle(csv_path)
    df.to_csv(str(csv_path).replace(".pkl", ".csv"), index=False)