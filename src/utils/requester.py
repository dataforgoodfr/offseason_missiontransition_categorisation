import requests
from bs4 import BeautifulSoup
import argparse
import numpy as np

URL_MT = "https://mission-transition-ecologique.beta.gouv.fr/api/temp/aids/"
URL_AT = "https://aides-territoires.beta.gouv.fr/api/aids/all/"
URL_AT_THEMES = "https://aides-territoires.beta.gouv.fr/api/themes/"
URL_MT_THEMES = "https://mission-transition-ecologique.beta.gouv.fr/api/environmental-topics/"

class Requester:
    def __init__(self) -> None:
        pass

    def get_aids(self) -> list:
        """
        Get all aids from the API.
        """
        response = requests.get(URL_AT)
        data = response.json()
        aids = data["results"]
        return aids

    def get_aid(self, aid_id: str) -> dict:
        """
        Get an aid from the API.
        """
        response = requests.get(URL_AT + aid_id)
        data = response.json()
        aid = data["results"]
        return aid

    def get_theme_aids(self, theme_id: str) -> list:
        """
        Get aids from the API with a theme.
        """
        response = requests.get(URL_AT_THEMES + theme_id)
        data = response.json()
        aids = data["results"]
        return aids

    def get_themes(self) -> list:
        """
        Get all themes from the API.
        """
        response = requests.get(URL_AT_THEMES)
        data = response.json()
        themes = data["results"]
        return themes

    def get_aids_mt(self) -> list:
        """
        Get all aids from the Mission Transition API.
        """
        response = requests.get(URL_MT)
        data = response.json()
        aids = data
        return aids

    def get_themes_mt(self) -> list:
        """
        Get all themes from the Mission Transition API.
        """
        response = requests.get(URL_MT_THEMES)
        data = response.json()
        themes = data
        return themes


elt = Requester()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get aids from the API.")
    # parser.add_argument('--help', metavar='N', type=int, help='', default=None)
    args = parser.parse_args()

    R = Requester()

    mt_aids = R.get_aids_mt()

    print("Colonnes aides mission transition: ", mt_aids[0].keys())
    l = R.get_aids()
    print("Colonnes aides aides territoires: ", l[0].keys())
    t = R.get_themes()
    print("Nombre de thématiques: ", len(t))
    print("Nombre de sous thématiques: ", np.sum([len(x["categories"]) for x in t]))
