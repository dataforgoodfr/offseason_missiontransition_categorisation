""" Baseline model for theme detection """
import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from utils.load_data import load_mt_aids
from data_preparation import DataPreparation


class BaselineClassifier:  # pylint: disable=too-few-public-methods
    """Class for topic detection based on term occurences
    Args:
        topic (str): topic to be detected
        tokens (list of str): tokens to be searched
    """
    def __init__(self, topic, tokens):
        self.topic = topic
        self.tokens = set(tokens)

    def predict(self, X):
        """ predict method for the baseline model
        Args:
            X (ndarray of list of strings): tokens
        """
        assert X.ndim == 1

        def has_common_token(tokens):
            return len(self.tokens.intersection(set(tokens))) > 0

        vfunc = np.vectorize(has_common_token)
        return vfunc(X)


class TopClassifier:  # pylint: disable=too-few-public-methods
    """ top classifier composed by multiple subclassifiers.
    Each subclassifier is dedicated to each topics
    """
    def __init__(self):
        topic_lexf = {
            "Ressources humaines":
            ["sensibilisation", "sensibiliser", "informer", "recruter",
             "recrutement", "coaching", "formation", "former", "formateur"],
            "Secteur bois":
            ["bois", "agroforesterie", "boisé", "forêt", "peuplier"],
            "Mobilité des employés":
            ['véhicule', 'mobilité', 'voiture', 'ferroviaire']
        }
        self.clfs = [BaselineClassifier(topic, tokens) for topic, tokens
                     in topic_lexf.items()]

    def scores(self, X, y):
        """ Return scores of the several baseline models
        Args:
            X (ndarray of list of str): vector of tokens
            y (list of str): vector of topic list (ground truth)
        Returns:
            dict: multiple metrics associated to each
               baseline classifier members of the class
        """
        res = {}
        for clf in self.clfs:
            y_pred = clf.predict(X)
            y_true = np.array([clf.topic in ii for ii in y])
            cur = {"recall": recall_score(y_true, y_pred),
                   "precision": precision_score(y_true, y_pred),
                   "false_pos": np.flatnonzero(y_pred & ~y_true),
                   "false_neg": np.flatnonzero(y_true & ~y_pred)
                   }
            res.update({clf.topic: cur})
        return res


if __name__ == "__main__":
    mt_aids = load_mt_aids()
    # remove aids with no labels
    mt_aids = mt_aids.loc[mt_aids.thematiques.apply(lambda x: 'NULL' not in x)]
    print(f"Found {len(mt_aids)} labelized aids")

    dtp = DataPreparation()
    X_raw = mt_aids.description.to_list()
    X_tokens = np.array(dtp.tokenize(X_raw), dtype=list)
    y_topics = mt_aids.thematiques.to_list()

    tclf = TopClassifier()
    scores = tclf.scores(X_tokens, y_topics)

    # display results
    df_res = pd.DataFrame.from_dict(scores, orient='index')
    df_res.false_neg = df_res.false_neg.apply(
        lambda x: mt_aids.index[x].to_list())
    df_res.false_pos = df_res.false_pos.apply(
        lambda x: mt_aids.index[x].to_list())
    print(df_res.drop(columns=["false_neg", "false_pos"]).to_string(
        float_format=lambda x: f"{x:.2f}"))

    # save results
    outfile = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           "../data/baseline_errors.csv")
    print(f"Saving results in {outfile}")
    df_res.to_csv(outfile)
