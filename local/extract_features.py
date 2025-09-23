import argparse
import functools
import pickle
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from tqdm import tqdm


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/wavlm-large", help="Huggingface model name")
    parser.add_argument("--dataset_csv", type=Path, help="Dataset to extract features")
    parser.add_argument("--split", default="both", choices=("train", "test", "both"), help="Dataset split to use")
    parser.add_argument("--output_path", type=Path, help="Output pkl path")
    parser.add_argument("--device", default="cpu", help="Device to infer, cpu or cuda:0 (gpu)")
    parser.add_argument("--layer_index", type=int, help="Layer index", default=-1)
    parser.add_argument("--store_raw_data", action="store_true", help="Store raw features")
    parser.add_argument("--pool", default="center", choices=("center", "average"), help="Pooling method")
    parser.add_argument("--slice", action="store_true", help="Slice audio")
    return parser.parse_args()


def _get_feat(row, feats, pool, stride_size):
    f = feats[row.audio_path]
    
    def _sec_to_index(t):
        i = int(t * 16000) // stride_size
        return np.clip(i, 0, len(f) - 1)

    if pool == "center":
        if "duration" in row:
            index = _sec_to_index((row["duration"]) / 2.0)
        else:
            index = _sec_to_index((row["min"] + row["max"]) / 2.0)
        return f[index]
    elif pool == "average":
        if "duration" in row:
            return f.mean(0)
        else:
            start_index = _sec_to_index(row["min"])
            end_index = _sec_to_index(row["max"])
            return f[start_index:end_index+1].mean(0)
    else:
        raise ValueError(f"Wrong parameter for pool: {pool}")

def _get_stride_size(model):
    if model in ("melspec", "mfcc"):
        return 512
    else:
        return 320

def _get_window_size(model):
    if model in ("melspec", "mfcc"):
        raise NotImplementedError("Window size is not implemented for melspec or mfcc")
    else:
        return 400


def _slice_with_min_window(x, min_i, max_i, window_size):
    n = len(x)
    current_size = max_i - min_i
    if current_size >= window_size:
        return x[min_i:max_i]

    # Compute the extra size needed
    extra = window_size - current_size
    left_expand = extra // 2
    right_expand = extra - left_expand

    # Expand min_i and max_i
    new_min_i = min_i - left_expand
    new_max_i = max_i + right_expand

    # Adjust if new_min_i is out of bounds
    if new_min_i < 0:
        shift = -new_min_i
        new_min_i = 0
        new_max_i = min(n, new_max_i + shift)

    # Adjust if new_max_i is out of bounds
    if new_max_i > n:
        shift = new_max_i - n
        new_max_i = n
        new_min_i = max(0, new_min_i - shift)

    assert (new_max_i - new_min_i) >= window_size
    return x[new_min_i:new_max_i]


def _infer(x, processor, model, args):
    x = processor(raw_speech=[x], sampling_rate=16000, padding=False, return_tensors="pt")

    if args.layer_index == -1:
        outputs = model(**{k: t.to(args.device) for k, t in x.items()})
        return outputs.last_hidden_state.cpu().detach().numpy()[0]
    else:
        outputs = model(output_hidden_states=True, **{k: t.to(args.device) for k, t in x.items()})
        return outputs.hidden_states[args.layer_index].cpu().detach().numpy()[0]

if __name__ == "__main__":
    args = _get_args()

    df = pd.read_csv(args.dataset_csv)
    if args.split != "both":
        df = df[df.split == args.split]

    raw_data_path = args.output_path.parent / f"{args.output_path.stem}.raw.pkl"

    if raw_data_path.exists():
        print("Using the cached features...")
        with open(raw_data_path, "rb") as f:
            data = pickle.load(f)
    else:
        print("Extracting features...")
        processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model).to(args.device)

        if args.slice:
            df["feat"] = None
            for path in tqdm(df.audio_path.unique()):
                x, _ = librosa.load(path, sr=16000, mono=True)
                for row in df[df.audio_path == path].itertuples():
                    sliced_x = _slice_with_min_window(x, int(row.min * 16000), int(row.max * 16000), _get_window_size(args.model))
                    df.at[row.Index, "feat"] = _infer(sliced_x, processor, model, args)
        else:
            data = {}
            for path in tqdm(df.audio_path.unique()):
                x, _ = librosa.load(path, sr=16000, mono=True)
                data[path] = _infer(x, processor, model, args)
            df["feat"] = df.apply(functools.partial(_get_feat, feats=data, pool=args.pool, stride_size=_get_stride_size(args.model)), axis=1)

    df.to_pickle(args.output_path)
