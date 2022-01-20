import ast
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (CamembertForSequenceClassification, CamembertTokenizerFast)

from modeling import CustomDataset

col_pred = "classes"
# col_pred = "subclasses"
nb_pred = 12
DEVICE = "cuda:0"
save_name = "model"
# nb_pred = 51


df = pd.read_csv("./processings.csv", sep="\t", encoding="utf-16")


df["text"] = df[["name", "aidDetails"]].apply(lambda x: ". ".join(x), axis=1)

df = df.reset_index(drop=True)
for i in range(df.shape[0]):
    for k in range(nb_pred):
        tmp = (
            df.loc[i, col_pred]
            .replace("[", "")
            .replace("]", "")
            .replace(",", "")
            .split(" ")
        )
        tmp = [int(elt) - 1 for elt in tmp if elt != ""]
        if k in tmp:
            df.loc[i, k] = 1
        else:
            df.loc[i, k] = 0

df = df.drop(["classes", "subclasses"], axis=1)
df["list"] = df[range(nb_pred)].apply(lambda x: [e for e in x], axis=1)

df.to_csv("./processings_train.csv", sep="\t", encoding="utf-16", index=False)


# Importing stock ml libraries

df = pd.read_csv("./processings_train.csv", sep="\t", encoding="utf-16")

df["list"] = df["list"].apply(lambda e: [int(x) for x in ast.literal_eval(e)])
new_df = df[["text", "list"]].copy()
new_df.head()
# Sections of config

# Defining some key variables that will be used later on in the training
MAX_LEN = 256
TRAIN_BATCH_SIZE = 1
VALID_BATCH_SIZE = 2
EPOCHS = 2
LEARNING_RATE = 1e-6
tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")

# Creating the dataset and dataloader for the neural network
train_size = 0.8
train_dataset = new_df.sample(frac=train_size, random_state=123)
test_dataset = new_df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)


print("Full dataset shape: {}".format(new_df.shape))
print("Train set shape: {}".format(train_dataset.shape))
print("Test set shape: {}".format(test_dataset.shape))

full_set = CustomDataset(new_df, tokenizer, MAX_LEN)
training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

train_params = {"batch_size": TRAIN_BATCH_SIZE, "shuffle": True, "num_workers": 0}

test_params = {"batch_size": VALID_BATCH_SIZE, "shuffle": True, "num_workers": 0}

full_loader = DataLoader(full_set, **train_params)
training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

pickle.dump(full_loader, open("../outputs/full_loader.pkl", "wb"))
pickle.dump(training_loader, open("../outputs/training_loader.pkl", "wb"))
pickle.dump(testing_loader, open("../outputs/testing_loader.pkl", "wb"))

model = CamembertForSequenceClassification.from_pretrained(
    "camembert-base", num_labels=nb_pred
)
model.to(DEVICE)


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs.logits, targets)


optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


def train(epoch, save_name="model"):
    model.train()
    best_loss = float("inf")
    for _, data in enumerate(training_loader, 0):
        ids = data["ids"].to(DEVICE, dtype=torch.long)
        mask = data["mask"].to(DEVICE, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(DEVICE, dtype=torch.long)
        targets = data["targets"].to(DEVICE, dtype=torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _ % 5000 == 0:
            print(f"Epoch: {epoch}, Loss:  {loss.item()}")
            if loss.item() < best_loss:
                model.save_pretrained("../outputs/{}/".format(save_name))
                best_loss = loss.item()
                print("Best model saved (epoch {})!".format(epoch))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


for epoch in range(EPOCHS):
    train(epoch, save_name=save_name)
