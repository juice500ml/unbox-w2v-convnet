# unbox-w2v-convnet

## How to run
```bash
# This will generate *.pkls to the pkls/ folder, saving convolutional features.
python3 save_embeddings.py

# Then, use jupyter to open visualization.ipynb
# It should generate plots without any problem!
jupyter lab

# FYI, you may try different hyperparameter grids via fixing configs.py
```

## Acknowledgements
- Implementation of linear CKA is from https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment
