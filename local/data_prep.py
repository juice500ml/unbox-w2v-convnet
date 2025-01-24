import glob
from pathlib import Path
from tqdm import tqdm
import argparse
import pandas as pd
import librosa
import os
import re


def get_spk_metadata(spk, df=pd.read_csv(Path(__file__).parent / "timit_speaker_metadata.csv")):
    row = df[df.speaker == spk].iloc[0]
    return row.sex, row.split


def get_phn_metadata(phn):
    match = re.match(r"([a-zA-Z]+)(\d+)$", phn)
    return match.group(1), match.group(2)


def get_continuum_id(spk, phn_start, phn_start_id, phn_end, phn_end_id):
    return f"{spk}-{phn_start}{phn_start_id}-{phn_end}{phn_end_id}"


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
def _prepare_authentic_timit(authentic_timit_path: Path):
    rows = []
    for p in tqdm(authentic_timit_path.glob("**/*.wav")):
        audio_path = str(p)
        filename = p.stem
        split = p.parent.name
        spk = p.stem.split("_")[1]
        _, split = get_spk_metadata(spk)
        phn = filename.split("_")[-1]
        duration = librosa.get_duration(path=audio_path)
        if duration < 0.025:
            continue
        rows.append({
            "audio_path": audio_path,
            "filename": filename,
            "speaker": spk,
            "phonemes": phn,
            "duration": duration,
            "split": split
        })
    return pd.DataFrame(rows)


## audio_path, filename, speaker, phoneme, duration
def _prepare_synthetic_timit(synthetic_timit_path: Path):
    rows = []
    for p in tqdm(synthetic_timit_path.glob("**/*.wav")):
        audio_path = str(p)
        spk, phn_start_and_id, phn_end_and_id, step = p.stem.split("_")
        phn_start, phn_start_id = get_phn_metadata(phn_start_and_id)
        phn_end, phn_end_id = get_phn_metadata(phn_end_and_id)
        spk_sex, split = get_spk_metadata(spk)

        duration = librosa.get_duration(path=audio_path)
        if duration < 0.025:
            continue
        rows.append({
            "audio_path": str(p),
            "speaker": spk,
            "phn_start": phn_start,
            "phn_start_id": phn_start_id,
            "phn_end": phn_end,
            "phn_end_id": phn_end_id,
            "continuum": get_continuum_id(spk, phn_start, phn_start_id, phn_end, phn_end_id),
            "step": int(step[4:]),
            "duration": duration,
            "split": split,
        })
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
        csv_path = args.output_path / f"{args.dataset_type}_{args.num_interpolation}.csv"
    else:
        csv_path = args.output_path / f"{args.dataset_type}.csv"
    df.to_csv(str(csv_path), index=False)
