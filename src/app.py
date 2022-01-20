from unittest import result
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import ast 
from modeling import CustomDataset
from torch.utils.data import DataLoader
from transformers import CamembertForSequenceClassification, CamembertTokenizerFast
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
list_blue_colors = ["#000091", "#072F5F", "#1261A0", "#3895D3", "#58CCED", "#6495ED", "#5A86D5", "#5077BE", 
                    "#4668A6", "#3C598E", "#324B77", "#1E2D47", "#141E2F", "#0A0F18"]
def summarize_text(text, model, tokenizer):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)

    outputs = model.generate(inputs["input_ids"], 
                            max_length=100, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

    return (tokenizer.decode(outputs[0]))

# embedding_model = SentenceTransformer("Sahajtomar/french_semantic")

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
df = pd.read_csv("./final.csv", sep="\t", encoding="utf-16")
df["Thématique"] = df["best_cat"].map(lambda x: LABELS[int(x)])

st.set_page_config(page_title="Analyse des aides publiques : Catégorisation et résumé", layout="wide")
st.title("Catégorisation des aides de Mission Transition Ecologique")
class_df = df.groupby("Thématique").count().reset_index()
class_df["Nombre d'aides"] = class_df["name"]
fig = px.pie(class_df, values="Nombre d'aides", names="Thématique", 
             color_discrete_sequence = list_blue_colors,
            #  title="Répartition des aides de Mission Transition Ecologique"
             )
st.plotly_chart(fig, use_container_width=True)

# Histogramme des aides par thématiques
st.subheader("Analyse des thématiques")
fig = px.bar(class_df, x="Thématique", y="Nombre d'aides", 
             color_discrete_sequence = ["#000091"],
            #  title="Nombre d'aides par thématique"
             )
st.plotly_chart(fig, use_container_width=True)

# Mots clés pour la recherche d'aides
# st.subheader("Recherche d'aides par mots clés")
# aid_text = st.text_input("Entrez le détail de l'aide")
# clicked = st.button("Résumer l'aide")

# Nuage de points des aides
st.subheader("Visualisation des aides")

fig = px.scatter(df, x="x", y="y", color="Thématique", hover_name="name", height=600, 
                #  title="Nuage des aides"
                 )
fig.update_layout(
     xaxis=dict(showgrid=False), 
     yaxis=dict(showgrid=False))
st.plotly_chart(fig, use_container_width=True)

# Camembert
st.subheader("Catégorisation et résumé des aides")
df["Thématiques"] = df["list"].map(lambda x: '\n - '.join(list(np.unique([LABELS[e] for e in ast.literal_eval(x)]))))
df["summary"] = df["summary"].apply(lambda x: x.replace("<pad>", "").replace("</s>", ""))
fig = go.Figure(data=[go.Table(
  header=dict(
    values=['<b>Titre</b>', '<b>Détails</b>', '<b>Résumé</b>' , '<b>Thématiques</b>'],
    line_color='black', fill_color='#000091',
    align='center',font=dict(color='white', size=16)
  ),
  cells=dict(
    values=[df.name.to_list(), df.aidDetails.to_list(), df.summary.to_list(), df["Thématiques"].to_list()],
    # line_color=[np.array(colors)[a],np.array(colors)[b], np.array(colors)[c]],
    # fill_color=[np.array(colors)[a],np.array(colors)[b], np.array(colors)[c]],
    line_color='#000091',
    fill_color='white',
    align='center', 
    font=dict(color='black', size=12)
    ))
])
fig.update_layout(height=600)
st.plotly_chart(fig, use_container_width=True)


# T5
st.subheader("Démonstration de la catégorisation d'une aide")
aid_text = st.text_input("Entrez le texte de l'aide")
clicked = st.button("Catégoriser l'aide")
if clicked:
    MAX_LEN = 256
    DEVICE = "cpu"
    model_path = "../notebooks/results/camembert_v2/"
    model = CamembertForSequenceClassification.from_pretrained(model_path).to(DEVICE)
    tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")
    clf_df = pd.DataFrame({"text": [aid_text], "list": [[0]*12]})

    full_set = CustomDataset(clf_df, tokenizer, MAX_LEN)

    full_loader = DataLoader(full_set, batch_size=1, shuffle=False)

    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for i, data in enumerate(full_loader, 0):
            ids = data["ids"].to(DEVICE, dtype=torch.long)
            mask = data["mask"].to(DEVICE, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(DEVICE, dtype=torch.long)
            targets = data["targets"].to(DEVICE, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(
                torch.sigmoid(outputs.logits).cpu().detach().numpy().tolist()
            )
            outputs = np.array(fin_outputs) >= 0.2
            result_label = list(np.array(LABELS)[np.where(outputs[0] == True)])
            print(result_label)
            print(len(result_label))
            if len(result_label) == 0:
                st.markdown("Aide non catégorisée")
            elif len(result_label) == 1:
                st.markdown("La catégorie rattachée à cette aide est : " + ' - '.join(result_label))
            else:
                st.markdown("Les catégories rattachées à cette aide sont : " + ' - '.join(result_label)) 

st.subheader("Démonstration du résumé d'une aide")
aid_text_summary = st.text_input("Entrez le détail de l'aide")
clicked_summary = st.button("Résumer l'aide")

if clicked_summary:
    t5_tokenizer = T5Tokenizer.from_pretrained("plguillou/t5-base-fr-sum-cnndm")
    t5_model = T5ForConditionalGeneration.from_pretrained("plguillou/t5-base-fr-sum-cnndm")

    summary = summarize_text(aid_text_summary, t5_model, t5_tokenizer)

    st.markdown("Résumé de l'aide : " + summary.replace("<pad>", "").replace("</s>", "")) 

