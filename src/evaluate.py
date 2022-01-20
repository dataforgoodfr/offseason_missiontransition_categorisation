from transformers import CamembertForSequenceClassification
import torch
from sklearn import metrics
import numpy as np 

import pickle
nb_pred = 12
DEVICE = "cuda:0"
model_path = "../notebooks/results/camembert_v2/"
model = CamembertForSequenceClassification.from_pretrained(
    model_path, num_labels=nb_pred
)
model = model.to(DEVICE)
testing_loader = pickle.load(open("../outputs/training_loader.pkl", "rb"))

def validation(model, ds):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in enumerate(ds, 0):
            ids = data["ids"].to(DEVICE, dtype=torch.long)
            mask = data["mask"].to(DEVICE, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(DEVICE, dtype=torch.long)
            targets = data["targets"].to(DEVICE, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(
                torch.sigmoid(outputs.logits).cpu().detach().numpy().tolist()
            )
    return fin_outputs, fin_targets


# for epoch in range(EPOCHS):
outputs, targets = validation(model, testing_loader)
outputs = np.array(outputs) >= 0.2
accuracy = metrics.accuracy_score(targets, outputs)
f1_score_micro = metrics.f1_score(targets, outputs, average="micro")
f1_score_macro = metrics.f1_score(targets, outputs, average="macro")
cm = metrics.multilabel_confusion_matrix(targets, outputs)
print(f"Accuracy Score = {accuracy}")
print(f"F1 Score (Micro) = {f1_score_micro}")
print(f"F1 Score (Macro) = {f1_score_macro}")
