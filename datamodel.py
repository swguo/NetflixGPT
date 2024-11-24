import pandas as pd
import os
from datasets import Dataset, DatasetDict

def prepare_data(data_path, output_dir):
    # Load and preprocess the dataset
    data = pd.read_csv(data_path, encoding="utf_8_sig")[['title', 'content']]
    dataset = Dataset.from_pandas(data)
    train_test_split = dataset.train_test_split(test_size=0.2)
    
    datasets = DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })

    # Save train and test datasets as CSV
    train_df = datasets['train'].to_pandas()
    test_df = datasets['test'].to_pandas()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    train_df.to_csv(f"{output_dir}/netflix_train_zhcn.csv", encoding="utf_8_sig", index=False)
    test_df.to_csv(f"{output_dir}/netflix_test_zhcn.csv", encoding="utf_8_sig", index=False)
    
    return datasets
