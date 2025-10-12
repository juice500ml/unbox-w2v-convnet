import glob
from pathlib import Path
from tqdm import tqdm
import argparse
import pandas as pd
import praatio.textgrid
import numpy as np
import librosa
import os
import re
from datasets import load_dataset


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
    parser.add_argument("--dataset_type", type=str, choices=["timit_authentic", "timit_synthetic", "timit_hf", "sylber", "voxangeles"])
    parser.add_argument("--num_interpolation", type=int, default=None)
    parser.add_argument("--output_path", type=Path, help="Output csv folder")
    return parser.parse_args()


def map_splits(df_path="spk_info.csv"):
    df = pd.read_csv(df_path)
    spk_split_map = dict(zip(list(df['speaker']), list(df['split'])))
    return spk_split_map


def _prepare_hf_timit(timit_path: Path):
    # adapted from https://github.com/juice500ml/acoustic-units-for-ood/blob/main/dataset_prep.py
    timit = load_dataset("timit_asr", data_dir=timit_path, trust_remote_code=True)

    rows = []
    for split in ["train", "test"]:
        for utterance in tqdm(timit[split]):
            audio_path = utterance["audio"]["path"]
            speaker = utterance["speaker_id"]
            alignment = utterance["phonetic_detail"]
            for phn, start, stop in zip(alignment["utterance"], alignment["start"], alignment["stop"]):
                ipa = {
                    # Stops
                    "b": "b",
                    "d": "d",
                    "g": "ɡ",
                    "p": "p",
                    "t": "t",
                    "k": "k",
                    "dx": "ɾ",
                    "q": "ʔ",

                    # Affricates
                    "jh": "d͡ʒ",
                    "ch": "t͡ʃ",

                    # Fricatives
                    "s": "s",
                    "sh": "ʃ",
                    "z": "z",
                    "zh": "ʒ",
                    "f": "f",
                    "th": "θ",
                    "v": "v",
                    "dh": "ð",

                    # Nasals
                    "m": "m",
                    "n": "n",
                    "ng": "ŋ",
                    "em": "m̩",
                    "en": "n̩",
                    "eng": "ŋ̍",
                    "nx": "ɾ̃",

                    # Semivowels and Glides
                    "l": "l",
                    "r": "ɹ",
                    "w": "w",
                    "y": "j",
                    "hh": "h",
                    "hv": "ɦ",
                    "el": "l̩",

                    # Vowels
                    "iy": "i",
                    "ih": "ɪ",
                    "eh": "ɛ",
                    "ae": "æ",
                    "aa": "ɑ",
                    "ah": "ʌ",
                    "ao": "ɔ",
                    "uh": "ʊ",
                    "uw": "u",
                    "ux": "ʉ",
                    "er": "ɝ",
                    "ax": "ə",
                    "ix": "ɨ",
                    "axr": "ɚ",
                    "ax-h": "ə̯",

                    # Diphthongs
                    # These are ignored for simplicity
                    "ey": None,
                    "aw": None,
                    "ay": None,
                    "oy": None,
                    "ow": None,

                    # Stops closures
                    # These are attached to their succeeding stop
                    "bcl": None,
                    "dcl": None,
                    "gcl": None,
                    "pcl": None,
                    "tcl": None,
                    "kcl": None,

                    # Non-speech
                    "pau": None,
                    "epi": None,
                    "h#": None,
                }[phn]
                if phn in ("b", "d", "g", "p", "t", "k", "jh", "ch"):
                    closure = {"b": "bcl", "d": "dcl", "g": "gcl", "p": "pcl", "t": "tcl", "k": "kcl", "jh": "dcl", "ch": "tcl"}[phn]
                    if rows[-1]["timit_phn"] == closure:
                        start = rows[-1]["min"] * 16000

                rows.append({
                    "audio_path": audio_path,
                    "speaker": speaker,
                    "min": start / 16000,
                    "max": stop / 16000,
                    "timit_phn": phn,
                    "ipa": ipa,
                    "split": split
                })

    return pd.DataFrame(rows)


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


def _prepare_sylber(root_path: Path):
    # data/interpolation_demo_samples_all/{}
    minimum_pairs = {'consonants-thin-thing': ('n', 'ŋ'), 'consonants-sick-sing': ('k', 'ŋ'), 'consonants-deep-deem': ('p', 'm'), 'consonants-gap-cap': ('g', 'k'), 'consonants-boon-moon': ('b', 'm'), 'consonants-kid-kin': ('d', 'n'), 'consonants-tip-sip': ('t', 's'), 'consonants-tell-sell': ('t', 's'), 'consonants-grid-grin': ('d', 'n'), 'consonants-zoo-sue': ('z', 's'), 'consonants-sip-seem': ('ɪ', 'i'), 'consonants-long-wrong': ('l', 'r'), 'consonants-lest-rest': ('l', 'r'), 'consonants-zeal-seal': ('z', 's'), 'consonants-dime-time': ('d', 't'), 'consonants-dine-nine': ('d', 'n2'), 'consonants-lock-rock': ('l', 'r'), 'consonants-vox-fox': ('v', 'f'), 'consonants-down-town': ('d', 't'), 'consonants-zip-sip': ('z', 's'), 'consonants-gauge-cage': ('g', 'k'), 'consonants-vine-fine': ('v', 'f'), 'consonants-bad-pad': ('b', 'p'), 'consonants-pig-ping': ('g', 'ŋ'), 'consonants-kin-king': ('n', 'ŋ'), 'consonants-bar-par': ('b', 'p'), 'consonants-bay-pay': ('b', 'p'), 'consonants-vill-fill': ('v', 'f'), 'consonants-vault-fault': ('v', 'f'), 'consonants-deal-kneel': ('d', 'n'), 'consonants-bean-mean': ('b', 'm'), 'consonants-ban-pan': ('b', 'p'), 'consonants-tale-sale': ('t', 's'), 'consonants-dull-null': ('d', 'n'), 'consonants-click-cling': ('k', 'ŋ'), 'consonants-dig-ding': ('g', 'ŋ'), 'consonants-deen-teen': ('d', 't'), 'consonants-rip-rim': ('p', 'm'), 'consonants-seed-seen': ('d', 'n'), 'consonants-dose-nose': ('d', 'n'), 'consonants-boast-most': ('b', 'm'), 'consonants-zig-sig': ('z', 's'), 'consonants-ball-mall': ('b', 'm'), 'consonants-dall-tall': ('d', 't'), 'consonants-bin-bing': ('n', 'ŋ'), 'consonants-goal-coal': ('g', 'k'), 'consonants-chit-chin': ('t', 'n'), 'consonants-trip-trim': ('p', 'm'), 'consonants-lane-rain': ('l', 'r'), 'consonants-sin-sing': ('n', 'ŋ'), 'consonants-tank-sank': ('t', 's'), 'consonants-gain-cane': ('g', 'k'), 'vowels-li-lu': ('i', 'u'), 'vowels-be-bi': ('ɛ', 'ɪ'), 'vowels-tu-tow': ('ʊ', 'oʊ'), 'vowels-su-sow': ('u', 'oʊ'), 'vowels-ta-te': ('ɑ', 'eɪ'), 'vowels-sa-sow': ('ɑ', 'oʊ'), 'vowels-sa-se': ('ɑ', 'eɪ'), 'vowels-ta-tow': ('ɑ', 'oʊ'), 'vowels-le-li': ('eɪ', 'i'), 'vowels-bi-bu': ('ɪ', 'ʊ'), 'vowels-la-le': ('ɑ', 'eɪ'), 'vowels-se-si': ('eɪ', 'i'), 'vowels-te-ti': ('eɪ', 'ɪ'), 'vowels-la-low': ('ɑ', 'oʊ'), 'vowels-ba-bow': ('ɑ', 'oʊ'), 'vowels-ti-tu': ('ɪ', 'ʊ'), 'vowels-lu-low': ('u', 'oʊ'), 'vowels-ba-be': ('ɑ', 'æ'), 'vowels-si-su': ('ɪ', 'ʊ'), 'vowels-bu-bow': ('u', 'oʊ')}

    rows = []
    for cv in ("consonants", "vowels"):
        for continuum in (root_path / cv).glob("*"):
            for i, audio_path in enumerate(sorted(list(continuum.glob("*.wav")))):
                start, end = continuum.name.split("-")
                continuum_id = f"{cv}-{start}-{end}"
                phn_start, phn_end = minimum_pairs[continuum_id]

                if i in (0, 10):
                    grid_path = root_path / f"segmentation/{audio_path.stem.replace('.0', '_0')}.TextGrid"
                    grid = praatio.textgrid.openTextgrid(grid_path, includeEmptyIntervals=False)
                    phn = phn_start if i == 0 else phn_end
                    phn_entry = [
                        p for p in grid.getTier("phones").entries
                        if p.label == phn
                    ][0]
                    phn_min = phn_entry.start
                    phn_max = phn_entry.end
                else:
                    phn_min, phn_max = None, None
                rows.append({
                    "audio_path": str(audio_path),
                    "word_start": start,
                    "word_end": end,
                    "phn_start": phn_start,
                    "phn_end": phn_end,
                    "min": phn_min,
                    "max": phn_max,
                    "continuum": continuum_id,
                    "step": i,
                })
    df = pd.DataFrame(rows)

    def _interpolate(group):
        group = group.sort_values("step")
        min0 = group.loc[group.index[0], "min"]
        min1 = group.loc[group.index[-1], "min"]
        max0 = group.loc[group.index[0], "max"]
        max1 = group.loc[group.index[-1], "max"]

        for i, min_val, max_val in zip(
            range(len(group)),
            np.linspace(min0, min1, len(group)),
            np.linspace(max0, max1, len(group)),
        ):
            group.loc[group.index[i], "min"] = min_val
            group.loc[group.index[i], "max"] = max_val
        return group

    df = df.groupby("continuum").apply(_interpolate, include_groups=False).reset_index(level=0)
    return df


def _prepare_voxangeles(root_path: Path):
    rows = []

    for path in (root_path / "data/audited_aligned").glob("**/*.TextGrid"):
        grid = praatio.textgrid.openTextgrid(path, includeEmptyIntervals=False)
        tier_name = next((x for x in grid.tierNames if x in ("phone", "phones", "Narrow")))
        for entry in grid.getTier(tier_name).entries:
            rows.append({
                "audio_path": str(path.with_suffix(".wav")),
                "min": entry.start,
                "max": entry.end,
                "phone": entry.label,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    args = _get_args()
    print(args)

    _prepare = {
        "timit_authentic": _prepare_authentic_timit,
        "timit_synthetic": _prepare_synthetic_timit,
        "timit_hf": _prepare_hf_timit,
        "sylber": _prepare_sylber,
        "voxangeles": _prepare_voxangeles,
    }[args.dataset_type]
    df = _prepare(args.dataset_path)

    os.makedirs(args.output_path, exist_ok=True)
    if args.num_interpolation is not None:
        csv_path = args.output_path / f"{args.dataset_type}_{args.num_interpolation}.csv"
    else:
        csv_path = args.output_path / f"{args.dataset_type}.csv"
    df.to_csv(str(csv_path), index=False)
    print("Stored to", csv_path)
