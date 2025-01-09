import numpy as np
from itertools import product


umap_configs = {
    "local": {
        "min_dist": 0.1,
        "n_neighbors": 15,
        "metric": "cosine",
    },
    "global": {
        "min_dist": 0.5,
        "n_neighbors": 100,
        "metric": "cosine",
    },
}


model_configs = {
    "base": { # 95M
        "name": "wav2vec 2.0 Base",
        "code": "facebook/wav2vec2-base",
        "shorthand": "Base",
    },
    "large": { # 317M
        "name": "wav2vec 2.0 Large",
        "code": "facebook/wav2vec2-large",
        "shorthand": "Large",
    },
    "xls": {
        "name": "wav2vec 2.0 XLS-R",
        "code": "facebook/wav2vec2-xls-r-300m",
        "shorthand": "XLS-R",
    },
}

vowels = {
    "ipa": list("ieɛaɑʌɤɯyøœɶɒɔou"),
    "f1": [240, 390, 610, 850, 750, 600, 460, 300, 235, 370, 585, 820, 700, 500, 360, 250],
    "f2": [2400, 2300, 1900, 1610, 940, 1170, 1310, 1390, 2100, 1900, 1710, 1530, 750, 700, 640, 595],
}

exp_configs = {}
exp_configs.update({
    f"window{w:.2f}": {
        "signal": "get_step_signal",
        "base_freq": 200,
        "sig_freq": 800,
        "sig_dur_ratio": w,
    }
    for w in [0.32, 0.16, 0.08, 0.04, 0.02, 0.01]
})
exp_configs.update({
    "f0_dist": {
        "signal": "get_signal",
        "freq": np.arange(100, 600, 100).tolist(),
    },
    "bias": {
        "signal": "get_signal",
        "freq": np.arange(100, 600, 100).tolist(),
        "bias": np.linspace(-0.5, 0.5, 5).tolist(),
        "mag": 0.5,
    },
    "f0": {
        "signal": "get_signal",
        "freq": np.linspace(10, 8000, 800).tolist(),
    },
    "f1f2": {
        "signal": "get_signal",
        "freq": list(product(
            [120],
            np.linspace(235, 850, 30),
            np.linspace(595, 2400, 30),
        )),
        "mag": (0.5, 0.35, 0.15),
    },
    "f1f2_detailed": {
        "signal": "get_signal",
        "freq": list(product(
            [120],
            np.linspace(100, 1000, 200),
            np.linspace(100, 3000, 600),
        )),
        "mag": (0.5, 0.35, 0.15),
    },
    "f0f1f2": {
        "signal": "get_signal",
        "freq": list(product(
            np.linspace(100, 225, 6),
            np.linspace(235, 850, 30),
            np.linspace(595, 2400, 30),
        )),
        "mag": (0.5, 0.35, 0.15),
    },
    # "f0f1f2_praat": {
    #     "signal": "get_signal_from_file",
    #     "freq": list(product(
    #         np.linspace(100, 225, 6),
    #         np.linspace(235, 850, 30),
    #         np.linspace(595, 2400, 30),
    #     )),
    # },
    # "f1f2_praat": {
    #     "signal": "get_signal_from_file",
    #     "freq": list(product(
    #         [125],
    #         np.linspace(235, 850, 30),
    #         np.linspace(595, 2400, 30),
    #     )),
    # },
    "w": {
        "signal": "get_signal",
        "freq": (100, 700),
        "mag": list(product(
            # Square root scaling
            np.square(np.linspace(np.sqrt(0.1), np.sqrt(0.5), 20)),
            np.square(np.linspace(np.sqrt(0.1), np.sqrt(0.5), 20)),
        ))
    },
    "diphthong": {
        "signal": "get_signal_diphthong",
        "f0": 120,
        "freq_start": list(zip(vowels["f1"], vowels["f2"])),
        "freq_finish": list(zip(vowels["f1"], vowels["f2"])),
        "mag": (0.5, 0.35, 0.15),
        "dur": 2.0,
        "margin": 0.1,
    }
})
