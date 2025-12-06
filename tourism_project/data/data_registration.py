from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
import pandas as pd
import os

# Load the tourism dataset from local file
df = pd.read_csv(google_drive_csv_path)

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nDataset info:\n{df.info()}")

# Convert to Hugging Face Dataset format
dataset = Dataset.from_pandas(df)

# Push to Hugging Face Hub
dataset.push_to_hub(
    "RahulSingh211/tourism_dataset",
    token=os.environ['HF_TOKEN']
)

print("\n Dataset successfully uploaded to Hugging Face Hub!")
print("Dataset URL: https://huggingface.co/datasets/RahulSingh211/tourism_dataset")
