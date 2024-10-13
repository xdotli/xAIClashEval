from openai import OpenAI
from os import getenv
from data.prompts import prompts
import pandas as pd
import json

df = pd.read_parquet("data/dataset/dataset.pqt")
print(df.head())


# gets API Key from environment variable OPENAI_API_KEY
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY"),
)

row = df[df['dataset'] == 'names'].iloc[200]
question = row['question']
context = row['context_original']
name = row['answer_original']

response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": 'json_object'},
    messages=[
        {"role": "system", "content": prompts.RAG_RESPONSE['names']},
        {"role": "user", "content": f"Name: {name}"}
    ],
    temperature=0,
    seed=0,
)

perturbations_dict = json.loads(response.choices[0].message.content)
print(perturbations_dict)
