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
    parser.add_argument("--output_path", type=Path, help="Output pkl path")
    parser.add_argument("--device", default="cpu", help="Device to infer, cpu or cuda:0 (gpu)")
    parser.add_argument("--framewise", action="store_true", help="store all the frames")
    parser.add_argument("--layer_index", type=int, help="Layer index", default=-1)
    parser.add_argument("--store_raw_data", action="store_true", help="Store raw features")
    parser.add_argument("--pool", default="center", choices=("center", "average"), help="Pooling method")
    return parser.parse_args()


def _get_feat(row, feats, pool, stride_size):
    f = feats[row.audio_path]
    
    def _sec_to_index(t):
        i = int(t * 16000) // stride_size
        return np.clip(i, 0, len(f) - 1)

    if pool == "center":
        index = _sec_to_index((row["duration"]) / 2.0)
        return f[index]
    elif pool == "average":
        return f.mean(0)
    else:
        raise ValueError(f"Wrong parameter for pool: {pool}")

def _get_feats(row, feats):
    return feats

def _get_stride_size(model):
    if model in ("melspec", "mfcc"):
        return 512
    else:
        return 320

if __name__ == "__main__":
    args = _get_args()

    df = pd.read_pickle(args.dataset_csv)

    raw_data_path = args.output_path.parent / f"{args.output_path.stem}.raw.pkl"

    if raw_data_path.exists():
        print("Using the cached features...")
        with open(raw_data_path, "rb") as f:
            data = pickle.load(f)
    else:
        print("Extracting features...")
        data = {}
        processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model).to(args.device)
        for path in tqdm(df.audio_path.unique()):
            x, _ = librosa.load(path, sr=16000, mono=True)
            x = processor(raw_speech=[x], sampling_rate=16000, padding=False, return_tensors="pt")

            if args.layer_index == -1:
                outputs = model(**{k: t.to(args.device) for k, t in x.items()})
                data[path] = outputs.last_hidden_state.cpu().detach().numpy()[0]
            else:
                outputs = model(output_hidden_states=True, **{k: t.to(args.device) for k, t in x.items()})
                data[path] = outputs.hidden_states[args.layer_index].cpu().detach().numpy()[0]
        if args.store_raw_data:
            with open(raw_data_path, "wb") as f:
                pickle.dump(data, f)

    stride_size = _get_stride_size(args.model)
    if args.framewise:
        _df = []
        for row in df.to_dict(orient="records"):
            for feat in _get_feats(row, data[row["audio_path"]]):
                _row = row.copy()
                _row["feat"] = feat
                _df.append(_row)
        df = pd.DataFrame(_df)
    else:
        df["feat"] = df.apply(functools.partial(_get_feat, feats=data, pool=args.pool, stride_size=stride_size), axis=1)
    df.to_pickle(args.output_path)