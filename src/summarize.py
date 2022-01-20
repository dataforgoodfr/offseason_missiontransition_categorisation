import pandas as pd 
import numpy as np 
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

df = pd.read_csv("./full_data.csv", sep="\t", encoding="utf-16")
print(df.head(5))
tokenizer = T5Tokenizer.from_pretrained("plguillou/t5-base-fr-sum-cnndm")
model = T5ForConditionalGeneration.from_pretrained("plguillou/t5-base-fr-sum-cnndm")

def summarize_text(text):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)

    outputs = model.generate(inputs["input_ids"], 
                            max_length=100, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

    return (tokenizer.decode(outputs[0]))

for i in tqdm(range(df.shape[0])):
    df.loc[i, "summary"] = summarize_text(df.loc[i, "aidDetails"])

df.to_csv("./final.csv", sep="\t", encoding="utf-16", index=False)