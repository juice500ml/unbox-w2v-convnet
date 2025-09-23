from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from cuml.cluster import KMeans
import cupy as cp  # GPU version of NumPy


def _kmeans(n_clusters, train_x, test_x):
    km = KMeans(n_clusters=n_clusters, max_iter=300, n_init='auto')
    km.fit(cp.asarray(train_x))
    return cp.asnumpy(km.predict(cp.asarray(test_x)))


def _prepare_data(df):
    index_to_vocab = dict(enumerate(df.phone.unique()))
    vocab_to_index = {v: i for i, v in index_to_vocab.items()}

    train_df = df[df.split == "train"]
    train_x = train_df.feat.tolist()
    train_y = train_df.phone.apply(vocab_to_index.get)

    test_df = df[df.split == "test"]
    test_x = test_df.feat.tolist()
    test_y = test_df.phone.apply(vocab_to_index.get)

    return train_x, train_y, test_x, test_y


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


    # phone purity
    p_p = sum(
        joint[y_star[j]][j]
        for j in range(len(uniq_cluster))
    )

    # cluster purity
    c_p = sum(
        joint[i][z_star[i]]
        for i in range(len(uniq_phone))
    )

    # mi
    mi = sum(
        joint[i][j] * np.log(joint[i][j] / (py[i] * pz[j]))
        for i, j in product(range(len(uniq_phone)), range(len(uniq_cluster)))
        if joint[i][j] > 0
    )

    # pnmi
    hy = sum(
        -py[i] * np.log(py[i])
        for i in range(len(uniq_phone))
        if py[i] > 0
    )
    pnmi = mi / hy

    return p_p, c_p, mi, pnmi


def get_kmeans_mi(train_x, test_x, test_y, n_clusters):
    return get_discrete_mi(test_y, _kmeans(n_clusters, train_x, test_x))


if __name__ == "__main__":
    fname = f"dsu_mi.csv"

    results = []
    for base, method in (
        ("wavlm", "21"),
        ("wavlm", "24"),

        ("wavlm", "textctc-commonphone-24"),
        ("wavlm", "phonectc-commonphone-24"),
        ("wavlm", "noblankctc-commonphone-24"),
        ("wavlm", "stepce-commonphone-24"),

        ("wavlm", "textctc-superblibri-24"),
        ("wavlm", "phonectc-superblibri-24"),
        ("wavlm", "noblankctc-superblibri-24"),

        ("wavlm", "phonectc-ftft-24"),
        ("wavlm", "noblankctc-ftft-24"),

        ("xlsr", "21"),
        ("xlsr", "24"),

        ("xlsr", "textctc-commonphone-24"),
        ("xlsr", "phonectc-commonphone-24"),
        ("xlsr", "noblankctc-commonphone-24"),
        ("xlsr", "stepce-commonphone-24"),

        ("xlsr", "textctc-superblibri-24"),
        ("xlsr", "phonectc-superblibri-24"),
        ("xlsr", "noblankctc-superblibri-24"),
    ):
        df = pd.read_pickle(f"data/timit-{base}-{method}.pkl")

        train_x, train_y, test_x, test_y = _prepare_data(df)
        p_p, c_p, mi, pnmi = get_kmeans_mi(train_x, test_x, test_y, n_clusters=38)
        results.append({"base": base, "method": method, "phone_purity": p_p, "cluster_purity": c_p, "mi": mi, "pnmi": pnmi})
        print(results[-1])
    pd.DataFrame(results).to_csv(fname, index=False)
