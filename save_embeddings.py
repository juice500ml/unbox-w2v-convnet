from configs import umap_configs, model_configs, exp_configs
from utils import get_signal, get_signal_from_file, get_feature, get_step_signal

from itertools import product
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoModelForPreTraining
import umap
import torch


for model_key, model_config in model_configs.items():
    print("Model:", model_config["name"])
    model = AutoModelForPreTraining.from_pretrained(
        model_config["code"]).to(torch.device("cpu"))

    for umap_key, umap_config in umap_configs.items():
        print("UMAP:", umap_key)
        reducer = umap.UMAP(**umap_config)

        for exp_key, exp_config in tqdm(exp_configs.items()):
            exp_config = exp_config.copy()

            path = Path(f"pkls/{model_key}_{umap_key}_{exp_key}.pkl")
            if path.exists():
                print(path, "exists. Skipping...")
                continue

            signal = exp_config.pop("signal")
            getters = {
                "get_signal": get_signal,
                "get_signal_from_file": get_signal_from_file,
            }

            if signal in getters.keys():
                iter_params = {k: v for k, v in exp_config.items() if isinstance(v, list)}
                const_params = {k: v for k, v in exp_config.items() if not isinstance(v, list)}

                feats = []
                raw_feats = []
                for iter_values in tqdm(product(*iter_params.values())):
                    param = const_params.copy()
                    param.update(dict(zip(iter_params.keys(), iter_values)))

                    sig = getters[signal](**param)
                    raw_feat = get_feature(model, sig)
                    feat = raw_feat.mean(0)

                    if exp_key == "f0_dist":
                        raw_feats.append(raw_feat)
                    feats.append(feat)

                feats = np.array(feats)
                raw_feats = np.array(raw_feats)
                embs = reducer.fit_transform(feats)
                pickle.dump({
                    "feats": feats,
                    "embs": embs,
                    "raw_feats": raw_feats,
                }, open(path, "wb"))

            if signal == "get_step_signal":
                base_feat = get_feature(model, get_signal(exp_config["base_freq"])).mean(0)
                sig_feat = get_feature(model, get_signal(exp_config["sig_freq"])).mean(0)

                step_sig, mask = get_step_signal(**exp_config)
                step_feat = get_feature(model, step_sig)

                embs = reducer.fit_transform(np.concatenate([
                    step_feat, base_feat[None, :], sig_feat[None, :]
                ], axis=0))
                pickle.dump({
                    "step_feat": step_feat,
                    "embs": embs,
                    "mask": mask,
                    "base_feat": base_feat,
                    "sig_feat": sig_feat,
                }, open(path, "wb"))
