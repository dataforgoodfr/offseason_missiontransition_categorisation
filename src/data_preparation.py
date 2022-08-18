""" data preparation for topic detection """
import spacy
from tqdm import tqdm
from bs4 import BeautifulSoup



class DataPreparation(): # pylint: disable=too-few-public-methods
    """data preparation for mission transition training set
    """

    def __init__(self):
        print("load spacy pipeline")
        try:
            self.nlp = spacy.load("fr_core_news_md", exclude=["ner"])
        except IOError:
            print("Try to run: python3 -m spacy download fr_core_news_md")
            raise

    def tokenize(self, docs, parse_html=True):
        """Scrap html, tokenize and lemmatize docs.
        also remove punctuation

        Args:
            docs (list of str): strings (html aid description typically)
        Returns:
            list of list of str (lems)
        """
        docs_parsed = docs
        if parse_html:
            docs_parsed = [
                BeautifulSoup(doc, features='html.parser').get_text()
                for doc in docs]

        disable_pos = ["PUNCT", "SYM"]
        with self.nlp.select_pipes(disable=["parser"]):
            docs_lems = [
                [w.lemma_ for w in doc if w.pos_ not in disable_pos]
                for doc in tqdm(self.nlp.pipe(docs_parsed), total=len(docs))]

        return docs_lems
