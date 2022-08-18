""" datasets from MT and AT loaded as DataFrames """
import pandas as pd
import numpy as np
from utils.requester import Requester


def load_df_mt_topics():
    """Return a well formated dataframe with MT topics and subtopics.

    Returns:
       pd.DataFrame: columns ["id", "thematique", "id_st", "sousThematique"]
    """

    mt_topics = Requester().get_themes_mt()

    dft = pd.json_normalize(mt_topics)
    dft = dft.explode("sousThematique").reset_index(drop=True).rename(
        columns={"nom": "thematique"})
    sous_thematiques = pd.DataFrame.from_records(
        dft.sousThematique.values).rename(
            columns={"nom": "sousThematique", "id": "id_st"})
    dft = pd.concat([dft[["id", "thematique"]], sous_thematiques], axis=1)

    # fix bug in Mobilité in API v3
    dft.loc[dft.sousThematique ==
            "Développer le réseau de distribution d'énergie",
            ["id", "thematique"]] = [8, "Production et distribution d'énergie"]

    return dft.sort_values(by=["id", "id_st"]).reset_index(drop=True)


def load_mt_aids():
    """Return a dataframe containing all mt aids."""
    aides_mt_js = Requester().get_aids_mt()

    aides_mt = pd.json_normalize(aides_mt_js).set_index("id")

    # fill empty sousThematique with 'NULL'
    empty_st = aides_mt.sousThematiques.apply(len) == 0
    aides_mt.loc[empty_st, "sousThematiques"] = pd.Series(
        [[{"nom": "NULL", "id": -1}]] * empty_st.sum(),
        index=empty_st[empty_st].index)

    # sousThematiques and id_st as list
    tmp_st = aides_mt[["sousThematiques"]].explode("sousThematiques")
    tmp_st = pd.DataFrame.from_records(
        tmp_st.sousThematiques.values, index=tmp_st.index).rename(
            columns={"nom": "sousThematiques", "id": "id_st"})
    tmp_st = pd.concat([
        tmp_st.groupby(tmp_st.index)["sousThematiques"].apply(list),
        tmp_st.groupby(tmp_st.index)["id_st"].apply(list)], axis=1)

    # join Thematique
    lut_topics = load_df_mt_topics().set_index(
        "sousThematique")['thematique'].to_dict()
    topic = tmp_st.sousThematiques.apply(
        lambda x: list(np.unique([lut_topics.get(st, 'NULL') for st in x])))

    return pd.concat([topic.rename("thematiques"), tmp_st,
                      aides_mt.drop(columns="sousThematiques")],
                     axis=1)
