
from sentence_transformers import SentenceTransformer
import pandas as pd 
from umap import UMAP

embedding_model = SentenceTransformer("Sahajtomar/french_semantic").to("cpu")


df = pd.read_csv("./processings_evaluated.csv", sep="\t", encoding="utf-16")
embeddings = embedding_model.encode(df.name.to_list())
umap_model = UMAP(n_neighbors=10, n_components=2 ,min_dist=0.1, metric="euclidean")

xy_emb = umap_model.fit_transform(embeddings)

df["embeddings"] = [str(list(embeddings[i, :].flatten())) for i in range(embeddings.shape[0])]
df["x"] = xy_emb[:,0]
df["y"] = xy_emb[:,1]
# print(df.head())

df.to_csv("./full_data.csv", sep="\t", encoding="utf-16", index=False)