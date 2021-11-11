import requests
from bs4 import BeautifulSoup

URL_MT = "https://mission-transition-ecologique.beta.gouv.fr/api/temp/aids/"
URL_AT = "https://aides-territoires.beta.gouv.fr/api/aids/all/"
URL_AT_THEMES = "https://aides-territoires.beta.gouv.fr/api/themes/"

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
    
elt = Requester()

print("### Mission Transition ### ")
mt_aids = elt.get_aids_mt()

print(len(mt_aids))
print(mt_aids[0].keys())


print("### Aides Territoires ###")
l = elt.get_aids()
t = elt.get_themes()
print(len(l))
print(l[0])
print(l[19].keys())
print(t[0])

print("####")
print(len(t))
print(t[0].keys())
