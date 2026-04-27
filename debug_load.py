import os
import pandas as pd

print("Starting data load...")
df = pd.read_csv("Data/normalized.csv")
print(f"Data loaded: shape = {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Data types:\n{df.dtypes}")
