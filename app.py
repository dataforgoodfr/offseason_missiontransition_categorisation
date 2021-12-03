import io
import sys 
import numpy as np 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torch import nn 
import uvicorn 
from transformers import CamembertTokenizerFast, CamembertForSequenceClassification

app = FastAPI()
LABELS = [
    "Evolution des procédés industriels",
    "Secteur Infrastructure & Déchéterie",
    "Innovation produits & services",
    "Gestion des bâtiments",
    "Production & distribution d'énergie",
    "Gestion des déchets",
    "Secteur Ville durable",
    "Secteur Agriculture & Zones rurales",
    "Mobilité des employés",
    "Secteur Eau & écosystèmes",
    "Ressources humaines",
    "Secteur Bois"
    ]

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")
model = CamembertForSequenceClassification.from_pretrained("./notebooks/results/camembert_v2/")
model = model.eval().cpu()

@app.get("/")
def root_route():
    return {"message": "Bonjour et bienvenue sur le serveur de catégorisation des aides de Mission Transition Ecologique"}

@app.get("/api/predict/{text}")
def predict(text):
    encoded_inputs = tokenizer.encode_plus(text, None, pad_to_max_length=True, max_length=512, return_token_type_ids=True, return_tensors="pt")
    outputs = model(**encoded_inputs)
    probabilities = nn.Sigmoid()(outputs.logits)
    predictions = list(np.array(probabilities.detach().cpu().numpy()>=0.2, dtype=int)[0])
    results = []
    print(predictions)
    for i, x in enumerate(predictions):
        if x == 1:
            results.append(LABELS[i])
    return {"predictions": results}



# uvicorn app:app --port=8080 --reload