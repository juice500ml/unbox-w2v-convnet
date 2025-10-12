# Mutual Information and Categorical Perception
Note. You need to copy all the content outside this folder.

## Prepare data
```bash
# TIMIT
envs/bin/python3 local/data_prep.py \
    --dataset_path datasets/timit \
    --dataset_type timit_hf \
    --output_path data

# Sylber continuums
envs/bin/python3 local/data_prep.py \
    --dataset_path datasets/interpolation_demo_samples_all \
    --dataset_type sylber \
    --output_path data
```

## Extract features
```bash
for layer in $(seq 0 24); do
    envs/bin/python3 local/extract_features.py \
        --model facebook/hubert-large-ll60k \
        --dataset_csv data/timit_hf.csv \
        --output_path data/timit-hubert-large-en-$layer.pkl \
        --device cuda:0 \
        --layer_index $layer \
        --pool average
done
```

## MI estimation
Note. the resulting CSV file is stored as `mi_full.csv` here.
```
envs/bin/python3 mutual_info.py
```

## Categorical perception
Run `geodesic.ipynb` and `ling_vs_ssl.ipynb`.