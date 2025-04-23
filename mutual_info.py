import pandas as pd
from sklearn.neural_network import MLPClassifier
import numpy as np
from scipy.stats import entropy
from itertools import product
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    from cuml.cluster import KMeans
    import cupy as cp  # GPU version of NumPy
    arr = cp.asarray(np.zeros(10))
    del arr
    print("Using GPU for kmeans")
    def _kmeans(n_clusters, train_x, test_x):
        km = KMeans(n_clusters=n_clusters, max_iter=300, n_init='auto')
        km.fit(cp.asarray(train_x))
        return cp.asnumpy(km.predict(cp.asarray(test_x)))
except:
    print("Using CPU for kmeans")
    from sklearn.cluster import KMeans
    def _kmeans(n_clusters, train_x, test_x):
        km = KMeans(n_clusters=n_clusters, random_state=42, verbose=1)
        km.fit(train_x)
        return km.predict(test_x)


def _prepare_data(df, label_type="phoneme"):
    assert label_type in ("phoneme", "speaker", "random")
    if label_type == "speaker":
        df.phone = df.speaker
        test_indices = df.sample(frac=0.2, random_state=42).index
        df.split = "train"
        df.loc[test_indices, "split"] = "test"
    if label_type == "random":
        df.phone = np.random.default_rng(seed=42).integers(0, 512, size=len(df))

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


def get_clf_mi(train_x, train_y, test_x, test_y, verbose=False):
    train_features = torch.tensor(train_x, dtype=torch.float32)
    train_labels = torch.tensor(train_y.to_list(), dtype=torch.long)  # for classification
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_features = torch.tensor(test_x, dtype=torch.float32)
    test_labels = torch.tensor(test_y.to_list(), dtype=torch.long)  # for classification
    test_dataset = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    input_dim = train_features.shape[1]
    num_classes = len(torch.unique(train_labels))  # adjust as needed
    epochs = 100
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Using", device, "for classifier")

    model = nn.Linear(input_dim, num_classes).to(device)
    model.reset_parameters()
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0)

    best_loss = 10000.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(batch_x.to(device))
            loss = criterion(outputs, batch_y.to(device)).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= i

        test_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x.to(device))
                test_loss += criterion(outputs, batch_y.to(device)).sum().item()
        test_loss /= len(test_y)
        if verbose:
            print(total_loss, test_loss)

        if best_loss > test_loss:
            best_loss = test_loss

    h_y = get_entropy(test_y)
    return h_y - best_loss


def get_kmeans_mi(train_x, test_x, test_y, n_clusters):
    return get_discrete_mi(test_y, _kmeans(n_clusters, train_x, test_x))


if __name__ == "__main__":
    fname = f"mi_full_km_lr.csv"

    results = []
    for dataset, model, layer, label_type in tqdm(product(
        ("timit", ),
        ("hubert-large-en", ),
        range(25),
        ("phoneme", "speaker", "random"),
    )):
        df = pd.read_pickle(f"data/{dataset}-{model}-{layer}.pkl")
        train_x, train_y, test_x, test_y = _prepare_data(df, label_type=label_type)
        for n_cluster in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            results.append({"dataset": dataset, "model": model, "layer": layer, "method": f"km_{n_cluster}_mi", "value": get_kmeans_mi(train_x, test_x, test_y, n_cluster), "speaker": by_speaker})
        results.append({"dataset": dataset, "model": model, "layer": layer, "method": "entropy", "value": get_entropy(test_y), "speaker": by_speaker})
        results.append({"dataset": dataset, "model": model, "layer": layer, "method": "clf_mi", "value": get_clf_mi(train_x, train_y, test_x, test_y), "speaker": by_speaker})
        pd.DataFrame(results).to_csv(fname, index=False)
