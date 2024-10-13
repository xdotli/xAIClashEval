import pandas as pd
import json
import numpy as np
import re

# from ..data.prompts import prompts
from data.prompts import prompts
from openai import OpenAI
client = OpenAI(api_key="YOUR_KEY_HERE")

df = pd.read_parquet("data/dataset/dataset.pqt")

row = df[df['dataset'] == 'names'].iloc[200]
question = row['question']
context = row['context_original']
name = row['answer_original']

print(f'Question: {question} \n')
print(f'Context: {context}')
