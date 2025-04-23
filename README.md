# Opening the Black Box of wav2vec Feature Encoder
Official implementation of the paper: https://arxiv.org/abs/2210.15386

## How to install
```bash
conda install pytorch==2.0.1 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers==4.35.0 pandas>2 librosa>0.10 numpy==1.23.5
python3 local/data_prep.py --dataset_path /data/user_data/eyeo2/data/CP/interpolated_vowels_13 --dataset_type timit_synthetic --num_interpolation 13 --output_path /home/kwanghec/unbox-w2v-convnet/data
python3 local/extract_features.py --model microsoft/wavlm-large --dataset_csv data/timit_synthetic_13.csv --output_path data/timit_synthetic_13_train_wavlm-large_24.pkl --device cuda:0 --layer_index 24 --pool average
```

## How to run
```bash
# This will generate *.pkls to the pkls/ folder, saving convolutional features.
python3 save_embeddings.py

# Then, use jupyter to open visualization.ipynb
# It should generate plots without any problem!
jupyter lab

# FYI, you may try different hyperparameter grids via fixing configs.py
```

## Praat experiments
You need a Praat installation beforehand to run the `praat/generate_audio.praat` via automated script, `praat/grid_audio.sh`.
You'll have to uncomment some of the code in `configs.py` before saving the embeddings.


## Acknowledgements
- Implementation of linear CKA is from https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment
