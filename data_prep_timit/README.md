
# TIMIT Dataset Preparation for Vowel Categorical Perception Tests

This repository provides scripts and instructions to prepare the TIMIT dataset for vowel categorical perception tests. The process includes phoneme segmentation, vowel selection, interpolation, and extraction of self-supervised learning (SSL) model representations.

## Prerequisites

- **TIMIT Dataset**: Download the TIMIT dataset from [LDC](https://catalog.ldc.upenn.edu/LDC93S1).


## Instructions

### Step 1: Segment Phonemes

Run the script to segment TIMIT audio into phoneme-level audio files.

```bash
python data_prep_timit/segment_phonoemes_timit.py --timit_dir TIMIT_DIR --output_dir OUTPUT_DIR
```

- `TIMIT_DIR`: Path to the downloaded TIMIT dataset.
- `OUTPUT_DIR`: Directory where segmented phoneme audio files will be saved.

Output: Audio files segmented at the phoneme level with filenames in the format:

```
"{dialect_region}_{speaker_id}_{utterance_id}_{phone_index}_{word}_{phone}.wav"
```

Example: `DR1_VMH0_SX386_9_do_z.wav`

### Step 2: Select Vowel Audios

Run the script to select vowel files for analysis, ensuring two criteria are met: 
(1) each file corresponds to one of the vowels 'iy,' 'ih,' 'eh,' 'ae,' 'ah,' 'uw,' 'uh,' 'ao,' or 'aa' (2) only speakers who produce all 9 vowels are selected.

```bash
python data_prep_timit/select_vowels.py --timit_dir TIMIT_DIR --phone_dir PHONE_DIR --dest_dir DEST_DIR
```

- `TIMIT_DIR`: Path to the TIMIT dataset (from Step 1).
- `PHONE_DIR`: Path to the segmented phoneme files (`OUTPUT_DIR` from Step 1).
- `DEST_DIR`: Directory where selected vowel audio files will be saved.

### Step 3: Interpolate Vowels

Generate interpolated vowel sounds for categorical perception tests.
For interpolation, we select the following pairs: [("iy", "ih"), ("ih", "eh"), ("eh", "ae"), ("iy", "uw"), ("ih", "uh"), ("eh", "ah"), ("ae", "aa"), ("uw", "uh"), ("uw", "ao"), ("ao","aa")].

```bash
python data_prep_timit/interpolate_vowels.py --base_path BASE_PATH --num_interpolation NUM_STEPS --output_dir OUTPUT_DIR
```

- `BASE_PATH`: Path to the directory containing selected vowel files (`DEST_DIR` from Step 2).
- `NUM_STEPS`: Number of interpolation steps.
- `OUTPUT_DIR`: Directory where interpolated vowel files will be saved.

### Step 4: Prepare Dataset Structure

Generate a structured dataset file for analysis.
To ensure compatibility with SSL feature extraction in the subsequent step, filter out audio files with a length shorter than 0.025 seconds.

```bash
python local/data_prep.py --dataset_path DATASET_PATH --dataset_type DATASET_TYPE --output_path OUTPUT_PATH --num_interpolation NUM_STEPS
```

- `DATASET_PATH`: Path to the vowel dataset (`DEST_DIR` from Step 2).
- `DATASET_TYPE`: Type of dataset (`timit_authentic` or `timit_synthetic`).
- `OUTPUT_PATH`: Path to save the structured dataset file (e.g., `.pkl`).
- `NUM_INTERPOLATION`: Number of interpolation steps (same as in Step 3).

Output: Audio information:

```
{
    "audio_path": audio_path,
    "filename": filename,
    "speaker": spk,
    "phonemes": phn,
    "duration": duration,
    "split": split
}
```

### Step 5: Extract SSL Representations

Extract SSL model representations from the vowel dataset.

```bash
python local/extract_features.py --dataset_csv DATASET_CSV --output_path OUTPUT_PATH --store_raw_data --framewise --pool
```

- `dataset_csv`: Path to the dataset file (output of Step 4, e.g., `timit_authentic.original.pkl` or `timit_synthetic.original.pkl`).
- `output_path`: Directory where the extracted representations will be saved.
- `store_raw_data` (optional): Store raw audio data along with features.
- `framewise` (optional): Extract frame-level features.
- `pool` (optional): Pooling strategy for features (`center` or `average`).


## Summary of Outputs

1. **Step 1**: Segmented phoneme-level audio files.
2. **Step 2**: Selected vowel audio files.
3. **Step 3**: Interpolated vowel audio files for categorical perception tests.
4. **Step 4**: Structured information about the audios.
5. **Step 5**: SSL model representations for vowel audios, dumped on Step 4.

