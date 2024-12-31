import csv
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

def map_sex(path):
    df = pd.read_csv(path, index_col=False)
    id_list = list(df["ID"])
    sex_list = list(df["Sex"])
    sex_dict = {id_list[i]: sex_list[i] for i in range(len(id_list))}
    return sex_dict

