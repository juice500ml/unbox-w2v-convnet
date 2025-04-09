import pandas as pd
from sklearn.neural_network import MLPClassifier
import numpy as np
from scipy.stats import entropy
from sklearn.cluster import KMeans
from itertools import product
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def _prepare_data(df, by_speaker=False):
    if by_speaker:
        df.phone = df.speaker
        test_indices = df.sample(frac=0.2, random_state=42).index
        df.split = "train"
        df.loc[test_indices, "split"] = "test"

    index_to_vocab = dict(enumerate(df.phone.unique()))
    vocab_to_index = {v: i for i, v in index_to_vocab.items()}

    train_df = df[df.split == "train"]
    train_x = train_df.feat.tolist()
    train_y = train_df.phone.apply(vocab_to_index.get)

    test_df = df[df.split == "test"]
    test_x = test_df.feat.tolist()
    test_y = test_df.phone.apply(vocab_to_index.get)

    return train_x, train_y, test_x, test_y


def get_entropy(test_y):
    _, p_y = np.unique(test_y, return_counts=True)
    p_y = p_y / len(test_y)
    h_y = entropy(p_y)
    return h_y


def get_discrete_mi(phone, cluster):
    uniq_phone = np.unique(phone)
    phone_to_idx = {p: i for i, p in enumerate(uniq_phone)}

    uniq_cluster = np.unique(cluster)
    cluster_to_idx = {c: i for i, c in enumerate(uniq_cluster)}

    # joint distrb
    joint = np.zeros((len(uniq_phone), len(uniq_cluster)))
    for p, c in zip(phone, cluster):
        joint[phone_to_idx[p]][cluster_to_idx[c]] += 1
    joint /= len(phone)

    pz = joint.sum(axis=0)
    py = joint.sum(axis=1)
    z_star = np.argmax(joint, axis=1)
    y_star = np.argmax(joint, axis=0)

    return sum(
        joint[i][j] * np.log(joint[i][j] / (py[i] * pz[j]))
        for i, j in product(range(len(uniq_phone)), range(len(uniq_cluster)))
        if joint[i][j] > 0
    )


def get_clf_mi(train_x, train_y, test_x, test_y):
    clf = MLPClassifier(random_state=42, hidden_layer_sizes=(), max_iter=500, verbose=1)
    clf.fit(train_x, train_y)

    h_y = get_entropy(test_y)
    return h_y + np.log(clf.predict_proba(test_x)[np.arange(len(test_y)), test_y]).mean()


def get_kmeans_mi(train_x, test_x, test_y, n_clusters):
    km = KMeans(n_clusters=n_clusters, random_state=42, verbose=1)
    km.fit(train_x)
    return get_discrete_mi(test_y, km.predict(test_x))


if __name__ == "__main__":
    fname = "mi_spk.csv"
    by_speaker = True

    results = []

    # for n_cluster in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
    #     for dataset in ("timit", ):
    #         for model in ("hubert_base-en", ):
    #             for layer in tqdm(range(12)):
    #                 df = pd.read_pickle(f"data/{dataset}-{model}-{layer}.pkl")
    #                 train_x, train_y, test_x, test_y = _prepare_data(df, by_speaker=by_speaker)
    #                 results.append({"dataset": dataset, "model": model, "layer": layer, "method": f"km_{n_cluster}_mi", "value": get_kmeans_mi(train_x, test_x, test_y, n_cluster)})
    #                 pd.DataFrame(results).to_csv(fname, index=False)

    for dataset in ("timit", ):
        for model in ("hubert_base-en", ):
            for layer in tqdm(range(12)):
                df = pd.read_pickle(f"data/{dataset}-{model}-{layer}.pkl")
                train_x, train_y, test_x, test_y = _prepare_data(df, by_speaker=by_speaker)

                results.append({"dataset": dataset, "model": model, "layer": layer, "method": "entropy", "value": get_entropy(test_y)})
                pd.DataFrame(results).to_csv(fname, index=False)
                results.append({"dataset": dataset, "model": model, "layer": layer, "method": "clf_mi", "value": get_clf_mi(train_x, train_y, test_x, test_y)})
                pd.DataFrame(results).to_csv(fname, index=False)
