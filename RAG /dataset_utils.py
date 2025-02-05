import pandas as pd
from datasets import Dataset
import pickle


def transform_dataframe_to_HFdataset(pickle_file: str, output_dir: str):
    with open(pickle_file, 'rb') as f:
        df = pd.read_pickle(pickle_file)
    
    hf_dataset = Dataset.from_pandas(df)
    hf_dataset.save_to_disk(output_dir)

if __name__ == "__main__":
    dataset = Dataset.load_from_disk("data/inter_HFdataset")
    dataset = dataset.select_columns(['union_diff', 'review2'])
    print(dataset)
    dataset.save_to_disk("data/inter_HFdataset-v2")


