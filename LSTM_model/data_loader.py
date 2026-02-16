import pandas as pd
import numpy as np
from datasets import load_dataset

def load_data():
  dataset = load_dataset("kemhug11/nasa-cmapss-rul")
  train_df = pd.DataFrame(dataset['train'])
  return train_df

