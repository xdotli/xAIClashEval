import pandas as pd
import json
import numpy as np
import re

# from ..data.prompts import prompts
from data.prompts import prompts
from openai import OpenAI
from os import getenv


client = OpenAI(
    api_key=getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)

df = pd.read_parquet("data/dataset/dataset.pqt")

row = df[df['dataset'] == 'names'].iloc[200]
question = row['question']
context = row['context_original']
name = row['answer_original']

print(f'Question: {question} \n')
print(f'Context: {context}')
