from transformers import CamembertForSequenceClassification, CamembertTokenizerFast
import pickle 
from torch.utils.data import DataLoader
import torch 
import pandas as pd 
from modeling import CustomDataset
import ast 
from tqdm import tqdm
import numpy as np 

MAX_LEN = 256
DEVICE = "cpu"
model_path = "../notebooks/results/camembert_v2/"
model = CamembertForSequenceClassification.from_pretrained(model_path).to(DEVICE)
tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")

df = pd.read_csv("./processings_train.csv", sep="\t", encoding="utf-16")
df["list"] = df["list"].apply(lambda e: [int(x) for x in ast.literal_eval(e)])

full_set = CustomDataset(df, tokenizer, MAX_LEN)

full_loader = DataLoader(full_set, batch_size=1, shuffle=False)

model.eval()
fin_targets = []
fin_outputs = []
with torch.no_grad():
    for i, data in tqdm(enumerate(full_loader, 0)):
        ids = data["ids"].to(DEVICE, dtype=torch.long)
        mask = data["mask"].to(DEVICE, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(DEVICE, dtype=torch.long)
        targets = data["targets"].to(DEVICE, dtype=torch.float)
        outputs = model(ids, mask, token_type_ids)
        fin_targets.extend(targets.cpu().detach().numpy().tolist())
        fin_outputs.extend(
            torch.sigmoid(outputs.logits).cpu().detach().numpy().tolist()
        )
        df.loc[i, "best_cat"] = np.argmax(outputs.logits.cpu().detach().numpy())
        
df.to_csv("./processings_evaluated.csv", sep="\t", encoding="utf-16", index=False)