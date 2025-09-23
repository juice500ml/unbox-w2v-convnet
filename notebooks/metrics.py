import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from scipy.special import softmax
import random
from tqdm import tqdm


def cos_sim_matrix(feats):
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    return np.dot(feats, feats.T)


def categorality_index(df, phn_start=None, phn_end=None):
    # Categorical Perception: A Groundwork for Deep Learning (Bonnasse-Gahot and Nadal 2022)

    _df = df[df.step == df.step.min()]
    feats = _df.feat.tolist()
    labels = np.array(_df.phn_start.tolist())

    mat = cos_sim_matrix(feats)
    triu = np.ones((len(labels), len(labels)), dtype=bool)
    triu = np.triu(triu) & (~np.eye(len(labels), dtype=bool))
    mask = (labels[None, :] == labels[:, None])

    return ks_2samp(mat[mask & triu].flatten(), mat[(~mask) & triu].flatten()).statistic


def discriminability_index(df, phn_start=None, phn_end=None): # smaller implies more CI
    # Sylber: Syllabic Embedding Representation of Speech from Raw Audio (Cheol Jun Cho et al. 2024)

    if phn_start is not None:
        assert phn_end is not None
        df = df[(df.phn_start == phn_start) & (df.phn_end == phn_end)]
    else:
        assert phn_end is None

    eps = np.finfo(df.iloc[0].feat[0]).eps
    dis = []

    for continuum, _df in df.groupby("continuum"):
        _df = _df.sort_values("step")
        mat = cos_sim_matrix(_df.feat.tolist())
        sim_l, sim_r = mat[0], mat[-1]
        prob_l = (sim_l - sim_l.min()) / (sim_l - sim_l.min() + sim_r - sim_r.min() + eps)

        area_l = np.cumsum(prob_l)
        area_r = np.cumsum((1.0 - prob_l[::-1]))
        l_disc = (area_l + area_r[::-1]) / len(prob_l)

        di = 1.0 - l_disc.max()
        dis.append(di)

    return np.array(dis).mean()

def softmax_discriminability_index(df, phn_start=None, phn_end=None): # smaller implies more CI
    if phn_start is not None:
        assert phn_end is not None
        df = df[(df.phn_start == phn_start) & (df.phn_end == phn_end)]
    else:
        assert phn_end is None

    dis = []
    for continuum, _df in df.groupby("continuum"):
        _df = _df.sort_values("step")
        mat = cos_sim_matrix(_df.feat.tolist()) * 10
        prob_l = softmax(mat[[1, -1], :], axis=0)[0]

        area_l = np.cumsum(prob_l)
        area_r = np.cumsum((1.0 - prob_l[::-1]))
        l_disc = (area_l + area_r[::-1]) / len(prob_l)

        di = 1.0 - l_disc.max()
        dis.append(di)

    return np.array(dis).mean()

def norm_softmax_discriminability_index(df, phn_start=None, phn_end=None): # smaller implies more CI
    if phn_start is not None:
        assert phn_end is not None
        df = df[(df.phn_start == phn_start) & (df.phn_end == phn_end)]
    else:
        assert phn_end is None

    dis = []
    for continuum, _df in df.groupby("continuum"):
        _df = _df.sort_values("step")
        mat = cos_sim_matrix(_df.feat.tolist()) * 10
        prob_l = softmax(mat[[1, -1], :], axis=0)[0]

        area_l = np.cumsum(prob_l)
        area_r = np.cumsum((1.0 - prob_l[::-1]))
        max_index = (area_l + area_r[::-1]).argmax()

        max_area_l = area_l[max_index] / (max_index + 1)
        max_area_r = area_r[::-1][max_index] / (len(area_r) - max_index)
        di = 1.0 - (max_area_l + max_area_r) / 2.0
        dis.append(di)
    return np.array(dis).mean()

def pairwise_categorality_index(df, phn_start=None, phn_end=None): # bigger implies more CI
    if phn_start is not None:
        assert phn_end is not None
        df = df[(df.phn_start == phn_start) & (df.phn_end == phn_end)]
    else:
        assert phn_end is None

    cis = []
    for continuum, _df in df.groupby("continuum"):
        _df = _df.sort_values("step")
        mat = cos_sim_matrix(_df.feat.tolist()) * 10
        prob_l = softmax(mat[[1, -1], :], axis=0)[0]

        area_l = np.cumsum(prob_l)[1:-1]
        area_r = np.cumsum((1.0 - prob_l[::-1]))[1:-1]
        max_index = (area_l + area_r[::-1]).argmax() + 1

        upper_mat = mat[:max_index, :max_index]
        lower_mat = mat[max_index + 1:, max_index + 1:]
        within_class = np.concatenate((upper_mat[np.triu_indices_from(upper_mat, k=1)], lower_mat[np.triu_indices_from(lower_mat, k=1)]))
        across_class = mat[:max_index, max_index + 1:].flatten()
        
        ci = ks_2samp(within_class, across_class).statistic
        cis.append(ci)
    return np.array(cis).mean()



if __name__ == "__main__":
    results = []

    for lang in ("ja", "as", "en", ):
        for layer in range(25):
            model = f"hubert-large-{lang}-{layer}"
            print(model)

            try:
                df = pd.read_pickle(f"data/timit_synthetic_13_test_{model}.pkl")
            except:
                print(model, "not loadable")
                continue
            for row in tqdm(df[["phn_start", "phn_end"]].drop_duplicates().itertuples()):
                
                results.append({
                    "model": model,
                    "lang": lang,
                    "layer": layer,
                    "sdi": softmax_discriminability_index(_df),
                    "di": discriminability_index(_df),
                    "nsdi": norm_softmax_discriminability_index(_df),
                    "pci": pairwise_categorality_index(_df),
                    "continuum": f"{row.phn_start}-{row.phn_end}",
                })
    results = pd.DataFrame(results)
    pd.DataFrame(results).to_csv(open("pci.csv", "w"), index=False)

    # results = []
    # for model in ("hubert-large-en-24", "hubert-large-ja-24", "hubert-large-as-24"):
    #     df = pd.read_pickle(f"data/timit_synthetic_13_test_{model}.pkl")
    #     results.append({
    #         "model": model.split("-")[2],
    #         "sdi": softmax_discriminability_index(df),
    #         "di": discriminability_index(df),
    #         "ci": categorality_index(df),
    #         "continuum": "all",        
    #     })
    # results = pd.DataFrame(results)
    # pd.DataFrame(results).to_csv(open("out.csv", "w"), index=False)
