import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def _normalize(emb):
    return emb / np.linalg.norm(emb)

def _random_sampler(df, phone, seed=42):
    l_indices = df[df.phone == phone].index.to_numpy()
    r_indices = df.index.to_numpy()
    np.random.default_rng(seed=seed).shuffle(r_indices)
    for l, r in zip(l_indices, r_indices):
        if l != r:
            yield l, r

def _speaker_sampler(df, phone, seed=42):
    _df = df[df.phone == phone]
    for spk in _df.speaker.unique():
        l_indices = _df[_df.speaker == spk].index.to_numpy()
        r_indices = df[df.speaker == spk].index.to_numpy()
        np.random.default_rng(seed=seed).shuffle(r_indices)
        for l, r in zip(l_indices, r_indices):
            if l != r:
                yield l, r

def _same_word_sampler(df, phone, seed=42):
    l_indices = df[df.phone == phone].index.to_numpy()
    r_indices = l_indices.copy()
    np.random.default_rng(seed=seed).shuffle(r_indices)
    for l, r in zip(l_indices, r_indices):
        if l != r:
            yield l, r

def _synth_word_sampler(df, phone, init, v1, v2, seed=42):
    def _get_indices(p):
        return df[df.phone == p].index.to_numpy().copy()
    def _shuffle(arr):
        np.random.default_rng(seed=seed).shuffle(arr)

    phone_i, init_i, v1_i, v2_i = _get_indices(phone), _get_indices(init), _get_indices(v1), _get_indices(v2)
    _shuffle(init_i), _shuffle(v1_i), _shuffle(v2_i)
    for phones in zip(phone_i, init_i, v1_i, v2_i):
        yield phones


# from collections import defaultdict
from tqdm import tqdm

# results = defaultdict(list)

# for layer in tqdm(range(25)):
#     df_full = pd.read_pickle(f"../feats/timit-wavlm-large-{layer}-audioslice.pkl")
#     df_full.feat = df_full.feat.apply(_normalize)
#     arpabet_to_ipa = {'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AX': 'ə', 'AXR': 'ɚ', 'AY': 'aɪ', 'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'IH': 'ɪ', 'IX': 'ɨ', 'IY': 'i', 'OW': 'oʊ', 'OY': 'ɔɪ', 'UH': 'ʊ', 'UW': 'u', 'UX': 'ʉ', 'B': 'b', 'CH': 't͡ʃ', 'D': 'd', 'DH': 'ð', 'DX': 'ɾ', 'EL': 'l̩', 'EM': 'm̩', 'EN': 'n̩', 'F': 'f', 'G': 'ɡ', 'HH': 'h', 'JH': 'd͡ʒ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NX': 'ɾ̃', 'NG': 'ŋ', 'Q': 'ʔ', 'P': 'p', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ', 'T': 't', 'WH': 'ʍ', 'TH': 'θ', 'V': 'v', 'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'}
#     df_full.allophone = df_full.allophone.apply(lambda a: arpabet_to_ipa.get(a.upper(), 'ignore'))
#     df_full.allophone.unique()
#     df_full.phone = df_full.allophone

#     synth_sets = []
#     # high-low
#     synth_sets += [('ɛ', 'ə', 'æ', 'ɑ'), ('æ', 'ɑ', 'ɛ', 'ə')]
#     # low-high
#     synth_sets += [('ə', 'ɛ', 'ɑ', 'æ'), ('ɑ', 'æ', 'ə', 'ɛ')]
#     # back-front (with tense not considered)
#     synth_sets += [('ɪ', 'ɛ', 'ɪ', 'æ'), ('ɪ', 'ɛ', 'i', 'æ'), ('ɪ', 'ɛ', 'ɛ', 'æ'), ('ɪ', 'ɛ', 'ə', 'ɑ'), ('i', 'ɛ', 'ɪ', 'æ'), ('i', 'ɛ', 'i', 'æ'), ('i', 'ɛ', 'ɛ', 'æ'), ('i', 'ɛ', 'ə', 'ɑ'), ('ɪ', 'æ', 'ɪ', 'ɛ'), ('ɪ', 'æ', 'i', 'ɛ'), ('ɪ', 'æ', 'ə', 'ɑ'), ('i', 'æ', 'ɪ', 'ɛ'), ('i', 'æ', 'i', 'ɛ'), ('i', 'æ', 'ə', 'ɑ'), ('ɛ', 'æ', 'ɪ', 'ɛ'), ('ɛ', 'æ', 'i', 'ɛ'), ('ɛ', 'æ', 'ə', 'ɑ'), ('ə', 'ɑ', 'ɪ', 'ɛ'), ('ə', 'ɑ', 'i', 'ɛ'), ('ə', 'ɑ', 'ɪ', 'æ'), ('ə', 'ɑ', 'i', 'æ'), ('ə', 'ɑ', 'ɛ', 'æ')]
#     # front-back (with tense not considered)
#     synth_sets += [('ɛ', 'ɪ', 'æ', 'ɪ'), ('ɛ', 'ɪ', 'æ', 'i'), ('ɛ', 'ɪ', 'æ', 'ɛ'), ('ɛ', 'ɪ', 'ɑ', 'ə'), ('ɛ', 'i', 'æ', 'ɪ'), ('ɛ', 'i', 'æ', 'i'), ('ɛ', 'i', 'æ', 'ɛ'), ('ɛ', 'i', 'ɑ', 'ə'), ('æ', 'ɪ', 'ɛ', 'ɪ'), ('æ', 'ɪ', 'ɛ', 'i'), ('æ', 'ɪ', 'ɑ', 'ə'), ('æ', 'i', 'ɛ', 'ɪ'), ('æ', 'i', 'ɛ', 'i'), ('æ', 'i', 'ɑ', 'ə'), ('æ', 'ɛ', 'ɛ', 'ɪ'), ('æ', 'ɛ', 'ɛ', 'i'), ('æ', 'ɛ', 'ɑ', 'ə'), ('ɑ', 'ə', 'ɛ', 'ɪ'), ('ɑ', 'ə', 'ɛ', 'i'), ('ɑ', 'ə', 'æ', 'ɪ'), ('ɑ', 'ə', 'æ', 'i'), ('ɑ', 'ə', 'æ', 'ɛ')]
#     # voiced-unvoiced
#     synth_sets += [('b', 'p', 'd', 't'), ('b', 'p', 'ɡ', 'k'), ('b', 'p', 'd͡ʒ', 't͡ʃ'), ('b', 'p', 'v', 'f'), ('b', 'p', 'z', 's'), ('b', 'p', 'ð', 'θ'), ('b', 'p', 'ʒ', 'ʃ'), ('d', 't', 'b', 'p'), ('d', 't', 'ɡ', 'k'), ('d', 't', 'd͡ʒ', 't͡ʃ'), ('d', 't', 'v', 'f'), ('d', 't', 'z', 's'), ('d', 't', 'ð', 'θ'), ('d', 't', 'ʒ', 'ʃ'), ('ɡ', 'k', 'b', 'p'), ('ɡ', 'k', 'd', 't'), ('ɡ', 'k', 'd͡ʒ', 't͡ʃ'), ('ɡ', 'k', 'v', 'f'), ('ɡ', 'k', 'z', 's'), ('ɡ', 'k', 'ð', 'θ'), ('ɡ', 'k', 'ʒ', 'ʃ'), ('d͡ʒ', 't͡ʃ', 'b', 'p'), ('d͡ʒ', 't͡ʃ', 'd', 't'), ('d͡ʒ', 't͡ʃ', 'ɡ', 'k'), ('d͡ʒ', 't͡ʃ', 'v', 'f'), ('d͡ʒ', 't͡ʃ', 'z', 's'), ('d͡ʒ', 't͡ʃ', 'ð', 'θ'), ('d͡ʒ', 't͡ʃ', 'ʒ', 'ʃ'), ('v', 'f', 'b', 'p'), ('v', 'f', 'd', 't'), ('v', 'f', 'ɡ', 'k'), ('v', 'f', 'd͡ʒ', 't͡ʃ'), ('v', 'f', 'z', 's'), ('v', 'f', 'ð', 'θ'), ('v', 'f', 'ʒ', 'ʃ'), ('z', 's', 'b', 'p'), ('z', 's', 'd', 't'), ('z', 's', 'ɡ', 'k'), ('z', 's', 'd͡ʒ', 't͡ʃ'), ('z', 's', 'v', 'f'), ('z', 's', 'ð', 'θ'), ('z', 's', 'ʒ', 'ʃ'), ('ð', 'θ', 'b', 'p'), ('ð', 'θ', 'd', 't'), ('ð', 'θ', 'ɡ', 'k'), ('ð', 'θ', 'd͡ʒ', 't͡ʃ'), ('ð', 'θ', 'v', 'f'), ('ð', 'θ', 'z', 's'), ('ð', 'θ', 'ʒ', 'ʃ'), ('ʒ', 'ʃ', 'b', 'p'), ('ʒ', 'ʃ', 'd', 't'), ('ʒ', 'ʃ', 'ɡ', 'k'), ('ʒ', 'ʃ', 'd͡ʒ', 't͡ʃ'), ('ʒ', 'ʃ', 'v', 'f'), ('ʒ', 'ʃ', 'z', 's'), ('ʒ', 'ʃ', 'ð', 'θ')]
#     # unvoiced-voiced
#     synth_sets += [('p', 'b', 't', 'd'), ('p', 'b', 'k', 'ɡ'), ('p', 'b', 't͡ʃ', 'd͡ʒ'), ('p', 'b', 'f', 'v'), ('p', 'b', 's', 'z'), ('p', 'b', 'θ', 'ð'), ('p', 'b', 'ʃ', 'ʒ'), ('t', 'd', 'p', 'b'), ('t', 'd', 'k', 'ɡ'), ('t', 'd', 't͡ʃ', 'd͡ʒ'), ('t', 'd', 'f', 'v'), ('t', 'd', 's', 'z'), ('t', 'd', 'θ', 'ð'), ('t', 'd', 'ʃ', 'ʒ'), ('k', 'ɡ', 'p', 'b'), ('k', 'ɡ', 't', 'd'), ('k', 'ɡ', 't͡ʃ', 'd͡ʒ'), ('k', 'ɡ', 'f', 'v'), ('k', 'ɡ', 's', 'z'), ('k', 'ɡ', 'θ', 'ð'), ('k', 'ɡ', 'ʃ', 'ʒ'), ('t͡ʃ', 'd͡ʒ', 'p', 'b'), ('t͡ʃ', 'd͡ʒ', 't', 'd'), ('t͡ʃ', 'd͡ʒ', 'k', 'ɡ'), ('t͡ʃ', 'd͡ʒ', 'f', 'v'), ('t͡ʃ', 'd͡ʒ', 's', 'z'), ('t͡ʃ', 'd͡ʒ', 'θ', 'ð'), ('t͡ʃ', 'd͡ʒ', 'ʃ', 'ʒ'), ('f', 'v', 'p', 'b'), ('f', 'v', 't', 'd'), ('f', 'v', 'k', 'ɡ'), ('f', 'v', 't͡ʃ', 'd͡ʒ'), ('f', 'v', 's', 'z'), ('f', 'v', 'θ', 'ð'), ('f', 'v', 'ʃ', 'ʒ'), ('s', 'z', 'p', 'b'), ('s', 'z', 't', 'd'), ('s', 'z', 'k', 'ɡ'), ('s', 'z', 't͡ʃ', 'd͡ʒ'), ('s', 'z', 'f', 'v'), ('s', 'z', 'θ', 'ð'), ('s', 'z', 'ʃ', 'ʒ'), ('θ', 'ð', 'p', 'b'), ('θ', 'ð', 't', 'd'), ('θ', 'ð', 'k', 'ɡ'), ('θ', 'ð', 't͡ʃ', 'd͡ʒ'), ('θ', 'ð', 'f', 'v'), ('θ', 'ð', 's', 'z'), ('θ', 'ð', 'ʃ', 'ʒ'), ('ʃ', 'ʒ', 'p', 'b'), ('ʃ', 'ʒ', 't', 'd'), ('ʃ', 'ʒ', 'k', 'ɡ'), ('ʃ', 'ʒ', 't͡ʃ', 'd͡ʒ'), ('ʃ', 'ʒ', 'f', 'v'), ('ʃ', 'ʒ', 's', 'z'), ('ʃ', 'ʒ', 'θ', 'ð')]
#     # fric-nonfric
#     synth_sets += [('s', 't', 'z', 'd'), ('s', 't', 'f', 'p'), ('s', 't', 'v', 'b'), ('z', 'd', 's', 't'), ('z', 'd', 'f', 'p'), ('z', 'd', 'v', 'b'), ('f', 'p', 's', 't'), ('f', 'p', 'z', 'd'), ('f', 'p', 'v', 'b'), ('v', 'b', 's', 't'), ('v', 'b', 'z', 'd'), ('v', 'b', 'f', 'p')]
#     # /ts/ → /t/
#     synth_sets += [('f', 'p', 't͡ʃ', 't'), ('v', 'b', 'd͡ʒ', 'ʒ')]
#     # nonfric-fric
#     synth_sets += [('t', 's', 'd', 'z'), ('t', 's', 'p', 'f'), ('t', 's', 'b', 'v'), ('d', 'z', 't', 's'), ('d', 'z', 'p', 'f'), ('d', 'z', 'b', 'v'), ('p', 'f', 't', 's'), ('p', 'f', 'd', 'z'), ('p', 'f', 'b', 'v'), ('b', 'v', 't', 's'), ('b', 'v', 'd', 'z'), ('b', 'v', 'p', 'f')]

#     # print("Rule              : Rnd.  Spk.  Same  | Synth")
#     for target, init, v1, v2 in synth_sets:
#         df = df_full[df_full.split == "test"].copy()
#         rand = np.mean([
#             np.dot(df.loc[l].feat, df.loc[r].feat)
#             for l, r in _random_sampler(df, target)
#         ])
#         spk = np.mean([
#             np.dot(df.loc[l].feat, df.loc[r].feat)
#             for l, r in _speaker_sampler(df, target)
#         ])
#         same = np.mean([
#             np.dot(df.loc[l].feat, df.loc[r].feat)
#             for l, r in _same_word_sampler(df, target)
#         ])

#         # MLE of von Mises–Fisher model
#         # We can also consider Fréchet mean (Karcher mean), but let's skip for now
#         synth = np.mean([
#             np.dot(df.loc[target_i].feat, _normalize(df.loc[init_i].feat + df.loc[v1_i].feat - df.loc[v2_i].feat))
#             for target_i, init_i, v1_i, v2_i in _synth_word_sampler(df, target, init, v1, v2)
#         ])
#         # print(f"{target:^3}<-{init:^3}+({v1:^3}-{v2:^3}): {rand:.3f} {spk:.3f} {same:.3f} | {synth:.3f}")
#         results[f"{target:^3}<-{init:^3}+({v1:^3}-{v2:^3})"].append(
#             {"rand": rand, "spk": spk, "same": same, "synth": synth}
#         )

# for key in results.keys():
#     results[key] = pd.DataFrame(results[key])

import pickle
import multiprocessing as mp


def process_synth_set(synth):
    """Worker for a single synth set. Relies on DF_TEST being available in the child via fork."""
    target, init, v1, v2 = synth
    df = DF_TEST.copy()

    def safe_mean(gen):
        vals = list(gen)
        return np.mean(vals) if len(vals) > 0 else np.nan

    rand = safe_mean(
        np.dot(df.loc[l].feat, df.loc[r].feat)
        for l, r in _random_sampler(df, target)
    )
    spk = safe_mean(
        np.dot(df.loc[l].feat, df.loc[r].feat)
        for l, r in _speaker_sampler(df, target)
    )
    same = safe_mean(
        np.dot(df.loc[l].feat, df.loc[r].feat)
        for l, r in _same_word_sampler(df, target)
    )

    synth = safe_mean(
        np.dot(df.loc[target_i].feat, _normalize(df.loc[init_i].feat + df.loc[v1_i].feat - df.loc[v2_i].feat))
        for target_i, init_i, v1_i, v2_i in _synth_word_sampler(df, target, init, v1, v2)
    )

    key = f"{target:^3}<-{init:^3}+({v1:^3}-{v2:^3})"
    return key, {"rand": rand, "spk": spk, "same": same, "synth": synth}


if __name__ == '__main__':
    # On Linux prefer 'fork' so child processes inherit the large dataframe without pickle
    try:
        mp.set_start_method('fork')
    except RuntimeError:
        # start method already set; assume it's OK
        pass

    # re-create results and run per-layer processing inside main
    from collections import defaultdict
    results = defaultdict(list)

    for layer in tqdm(range(25)):
        df_full = pd.read_pickle(f"../feats/timit-wavlm-large-{layer}-featslice.pkl")
        df_full.feat = df_full.feat.apply(_normalize)
        arpabet_to_ipa = {'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AX': 'ə', 'AXR': 'ɚ', 'AY': 'aɪ', 'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'IH': 'ɪ', 'IX': 'ɨ', 'IY': 'i', 'OW': 'oʊ', 'OY': 'ɔɪ', 'UH': 'ʊ', 'UW': 'u', 'UX': 'ʉ', 'B': 'b', 'CH': 't͡ʃ', 'D': 'd', 'DH': 'ð', 'DX': 'ɾ', 'EL': 'l̩', 'EM': 'm̩', 'EN': 'n̩', 'F': 'f', 'G': 'ɡ', 'HH': 'h', 'JH': 'd͡ʒ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NX': 'ɾ̃', 'NG': 'ŋ', 'Q': 'ʔ', 'P': 'p', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ', 'T': 't', 'WH': 'ʍ', 'TH': 'θ', 'V': 'v', 'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'}
        df_full.allophone = df_full.allophone.apply(lambda a: arpabet_to_ipa.get(a.upper(), 'ignore'))
        df_full.allophone.unique()
        df_full.phone = df_full.allophone

        # Work only on the test split for similarity computations
        DF_TEST = df_full[df_full.split == "test"].copy()

        synth_sets = []
        # high-low
        synth_sets += [('ɛ', 'ə', 'æ', 'ɑ'), ('æ', 'ɑ', 'ɛ', 'ə')]
        # low-high
        synth_sets += [('ə', 'ɛ', 'ɑ', 'æ'), ('ɑ', 'æ', 'ə', 'ɛ')]
        # back-front (with tense not considered)
        synth_sets += [('ɪ', 'ɛ', 'ɪ', 'æ'), ('ɪ', 'ɛ', 'i', 'æ'), ('ɪ', 'ɛ', 'ɛ', 'æ'), ('ɪ', 'ɛ', 'ə', 'ɑ'), ('i', 'ɛ', 'ɪ', 'æ'), ('i', 'ɛ', 'i', 'æ'), ('i', 'ɛ', 'ɛ', 'æ'), ('i', 'ɛ', 'ə', 'ɑ'), ('ɪ', 'æ', 'ɪ', 'ɛ'), ('ɪ', 'æ', 'i', 'ɛ'), ('ɪ', 'æ', 'ə', 'ɑ'), ('i', 'æ', 'ɪ', 'ɛ'), ('i', 'æ', 'i', 'ɛ'), ('i', 'æ', 'ə', 'ɑ'), ('ɛ', 'æ', 'ɪ', 'ɛ'), ('ɛ', 'æ', 'i', 'ɛ'), ('ɛ', 'æ', 'ə', 'ɑ'), ('ə', 'ɑ', 'ɪ', 'ɛ'), ('ə', 'ɑ', 'i', 'ɛ'), ('ə', 'ɑ', 'ɪ', 'æ'), ('ə', 'ɑ', 'i', 'æ'), ('ə', 'ɑ', 'ɛ', 'æ')]
        # front-back (with tense not considered)
        synth_sets += [('ɛ', 'ɪ', 'æ', 'ɪ'), ('ɛ', 'ɪ', 'æ', 'i'), ('ɛ', 'ɪ', 'æ', 'ɛ'), ('ɛ', 'ɪ', 'ɑ', 'ə'), ('ɛ', 'i', 'æ', 'ɪ'), ('ɛ', 'i', 'æ', 'i'), ('ɛ', 'i', 'æ', 'ɛ'), ('ɛ', 'i', 'ɑ', 'ə'), ('æ', 'ɪ', 'ɛ', 'ɪ'), ('æ', 'ɪ', 'ɛ', 'i'), ('æ', 'ɪ', 'ɑ', 'ə'), ('æ', 'i', 'ɛ', 'ɪ'), ('æ', 'i', 'ɛ', 'i'), ('æ', 'i', 'ɑ', 'ə'), ('æ', 'ɛ', 'ɛ', 'ɪ'), ('æ', 'ɛ', 'ɛ', 'i'), ('æ', 'ɛ', 'ɑ', 'ə'), ('ɑ', 'ə', 'ɛ', 'ɪ'), ('ɑ', 'ə', 'ɛ', 'i'), ('ɑ', 'ə', 'æ', 'ɪ'), ('ɑ', 'ə', 'æ', 'i'), ('ɑ', 'ə', 'æ', 'ɛ')]
        # voiced-unvoiced
        synth_sets += [('b', 'p', 'd', 't'), ('b', 'p', 'ɡ', 'k'), ('b', 'p', 'd͡ʒ', 't͡ʃ'), ('b', 'p', 'v', 'f'), ('b', 'p', 'z', 's'), ('b', 'p', 'ð', 'θ'), ('b', 'p', 'ʒ', 'ʃ'), ('d', 't', 'b', 'p'), ('d', 't', 'ɡ', 'k'), ('d', 't', 'd͡ʒ', 't͡ʃ'), ('d', 't', 'v', 'f'), ('d', 't', 'z', 's'), ('d', 't', 'ð', 'θ'), ('d', 't', 'ʒ', 'ʃ'), ('ɡ', 'k', 'b', 'p'), ('ɡ', 'k', 'd', 't'), ('ɡ', 'k', 'd͡ʒ', 't͡ʃ'), ('ɡ', 'k', 'v', 'f'), ('ɡ', 'k', 'z', 's'), ('ɡ', 'k', 'ð', 'θ'), ('ɡ', 'k', 'ʒ', 'ʃ'), ('d͡ʒ', 't͡ʃ', 'b', 'p'), ('d͡ʒ', 't͡ʃ', 'd', 't'), ('d͡ʒ', 't͡ʃ', 'ɡ', 'k'), ('d͡ʒ', 't͡ʃ', 'v', 'f'), ('d͡ʒ', 't͡ʃ', 'z', 's'), ('d͡ʒ', 't͡ʃ', 'ð', 'θ'), ('d͡ʒ', 't͡ʃ', 'ʒ', 'ʃ'), ('v', 'f', 'b', 'p'), ('v', 'f', 'd', 't'), ('v', 'f', 'ɡ', 'k'), ('v', 'f', 'd͡ʒ', 't͡ʃ'), ('v', 'f', 'z', 's'), ('v', 'f', 'ð', 'θ'), ('v', 'f', 'ʒ', 'ʃ'), ('z', 's', 'b', 'p'), ('z', 's', 'd', 't'), ('z', 's', 'ɡ', 'k'), ('z', 's', 'd͡ʒ', 't͡ʃ'), ('z', 's', 'v', 'f'), ('z', 's', 'ð', 'θ'), ('z', 's', 'ʒ', 'ʃ'), ('ð', 'θ', 'b', 'p'), ('ð', 'θ', 'd', 't'), ('ð', 'θ', 'ɡ', 'k'), ('ð', 'θ', 'd͡ʒ', 't͡ʃ'), ('ð', 'θ', 'v', 'f'), ('ð', 'θ', 'z', 's'), ('ð', 'θ', 'ʒ', 'ʃ'), ('ʒ', 'ʃ', 'b', 'p'), ('ʒ', 'ʃ', 'd', 't'), ('ʒ', 'ʃ', 'ɡ', 'k'), ('ʒ', 'ʃ', 'd͡ʒ', 't͡ʃ'), ('ʒ', 'ʃ', 'v', 'f'), ('ʒ', 'ʃ', 'z', 's'), ('ʒ', 'ʃ', 'ð', 'θ')]
        # unvoiced-voiced
        synth_sets += [('p', 'b', 't', 'd'), ('p', 'b', 'k', 'ɡ'), ('p', 'b', 't͡ʃ', 'd͡ʒ'), ('p', 'b', 'f', 'v'), ('p', 'b', 's', 'z'), ('p', 'b', 'θ', 'ð'), ('p', 'b', 'ʃ', 'ʒ'), ('t', 'd', 'p', 'b'), ('t', 'd', 'k', 'ɡ'), ('t', 'd', 't͡ʃ', 'd͡ʒ'), ('t', 'd', 'f', 'v'), ('t', 'd', 's', 'z'), ('t', 'd', 'θ', 'ð'), ('t', 'd', 'ʃ', 'ʒ'), ('k', 'ɡ', 'p', 'b'), ('k', 'ɡ', 't', 'd'), ('k', 'ɡ', 't͡ʃ', 'd͡ʒ'), ('k', 'ɡ', 'f', 'v'), ('k', 'ɡ', 's', 'z'), ('k', 'ɡ', 'θ', 'ð'), ('k', 'ɡ', 'ʃ', 'ʒ'), ('t͡ʃ', 'd͡ʒ', 'p', 'b'), ('t͡ʃ', 'd͡ʒ', 't', 'd'), ('t͡ʃ', 'd͡ʒ', 'k', 'ɡ'), ('t͡ʃ', 'd͡ʒ', 'f', 'v'), ('t͡ʃ', 'd͡ʒ', 's', 'z'), ('t͡ʃ', 'd͡ʒ', 'θ', 'ð'), ('t͡ʃ', 'd͡ʒ', 'ʃ', 'ʒ'), ('f', 'v', 'p', 'b'), ('f', 'v', 't', 'd'), ('f', 'v', 'k', 'ɡ'), ('f', 'v', 't͡ʃ', 'd͡ʒ'), ('f', 'v', 's', 'z'), ('f', 'v', 'θ', 'ð'), ('f', 'v', 'ʃ', 'ʒ'), ('s', 'z', 'p', 'b'), ('s', 'z', 't', 'd'), ('s', 'z', 'k', 'ɡ'), ('s', 'z', 't͡ʃ', 'd͡ʒ'), ('s', 'z', 'f', 'v'), ('s', 'z', 'θ', 'ð'), ('s', 'z', 'ʃ', 'ʒ'), ('θ', 'ð', 'p', 'b'), ('θ', 'ð', 't', 'd'), ('θ', 'ð', 'k', 'ɡ'), ('θ', 'ð', 't͡ʃ', 'd͡ʒ'), ('θ', 'ð', 'f', 'v'), ('θ', 'ð', 's', 'z'), ('θ', 'ð', 'ʃ', 'ʒ'), ('ʃ', 'ʒ', 'p', 'b'), ('ʃ', 'ʒ', 't', 'd'), ('ʃ', 'ʒ', 'k', 'ɡ'), ('ʃ', 'ʒ', 't͡ʃ', 'd͡ʒ'), ('ʃ', 'ʒ', 'f', 'v'), ('ʃ', 'ʒ', 's', 'z'), ('ʃ', 'ʒ', 'θ', 'ð')]
        # fric-nonfric
        synth_sets += [('s', 't', 'z', 'd'), ('s', 't', 'f', 'p'), ('s', 't', 'v', 'b'), ('z', 'd', 's', 't'), ('z', 'd', 'f', 'p'), ('z', 'd', 'v', 'b'), ('f', 'p', 's', 't'), ('f', 'p', 'z', 'd'), ('f', 'p', 'v', 'b'), ('v', 'b', 's', 't'), ('v', 'b', 'z', 'd'), ('v', 'b', 'f', 'p')]
        # /ts/ → /t/
        synth_sets += [('f', 'p', 't͡ʃ', 't'), ('v', 'b', 'd͡ʒ', 'ʒ')]
        # nonfric-fric
        synth_sets += [('t', 's', 'd', 'z'), ('t', 's', 'p', 'f'), ('t', 's', 'b', 'v'), ('d', 'z', 't', 's'), ('d', 'z', 'p', 'f'), ('d', 'z', 'b', 'v'), ('p', 'f', 't', 's'), ('p', 'f', 'd', 'z'), ('p', 'f', 'b', 'v'), ('b', 'v', 't', 's'), ('b', 'v', 'd', 'z'), ('b', 'v', 'p', 'f')]

        # Create a Pool after DF_TEST is assigned so child processes inherit DF_TEST via fork
        ctx = mp.get_context('fork')
        with ctx.Pool() as pool:
            for key, res in tqdm(pool.imap_unordered(process_synth_set, synth_sets), total=len(synth_sets)):
                results[key].append(res)

    # convert lists to DataFrames and save
    for key in list(results.keys()):
        results[key] = pd.DataFrame(results[key])

    with open("synthesis_featslice_results.pkl", "wb") as f:
        pickle.dump(results, f)
