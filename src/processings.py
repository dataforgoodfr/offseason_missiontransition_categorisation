from utils.requester import Requester
import numpy as np
import pandas as pd
import seaborn as sns
from bs4 import BeautifulSoup


def toStr(x):
    if pd.isna(x):
        return ""
    else:
        x = str(x).replace("\n", " ").strip()
        return x


R = Requester()
A = R.get_aids_mt()
T = R.get_themes_mt()

COLS = [
    "sourceId",
    "name",
    "perimeter",
    "aidDetails",
    "eligibility",
    "fundingSourceUrl",
    "applicationEndDate",
    "applicationUrl",
    "slug",
    "environmentalTopics",
    "funder",
    "fundingTypes",
    "regions",
    "contactGuidelines",
    "subventionRateUpperBound",
    "subventionRateLowerBound",
    "loanAmount",
    "applicationStartDate",
    "projectExamples",
    "directAccess",
    "types",
    "id",
]
df = pd.DataFrame(columns=COLS)

for i in range(len(A)):
    df = df.append(pd.DataFrame([[A[i][k] for k in COLS]], columns=COLS))

print(df.shape)
df["name"] = df["name"].apply(lambda x: toStr(x))
df["aidDetails"] = df["aidDetails"].apply(lambda x: BeautifulSoup(toStr(x)).get_text())
df["eligibility"] = df["eligibility"].apply(
    lambda x: BeautifulSoup(toStr(x)).get_text()
)

df = df[["sourceId", "id", "name", "aidDetails", "environmentalTopics"]].reset_index(
    drop=True
)


df["subclasses"] = df["environmentalTopics"].apply(lambda x: [i["id"] for i in x])


TF = pd.DataFrame(columns=T[0].keys())
for i in range(len(T)):
    TF = TF.append(T[i], ignore_index=True)


TF["subclasses"] = TF["environmentalTopics"].apply(lambda x: [i["id"] for i in x])

subclasses2classes = {}
for i in range(TF.shape[0]):
    for l in TF.loc[i, "subclasses"]:
        subclasses2classes[l] = TF.loc[i, "id"]


df["classes"] = df["subclasses"].apply(
    lambda x: np.unique([subclasses2classes[i] for i in x])
)


print(df.shape)
df.to_csv("./processings.csv", sep="\t", encoding="utf-16", index=False)
