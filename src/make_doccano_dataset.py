"""Build json dataset for doccano"""
import re
import json
from addict import Dict
from markdownify import markdownify
from utils.requester import Requester


def textify_html(html):
    """Generate raw readable text from html
    Args:
        html (string): html
    """
    html1 = html.replace("\r\n", "")  # remove newline in windows

    # remove strong case because won't work well with mardownify
    html2 = re.sub("[ \\n]{1,}<strong>[ \\n]{1,}", " ", html1)
    html2 = re.sub("[ \\n]{1,}</strong>[ \\n]{1,}", " ", html2)

    # use mardownify lib to transform into text
    mkd = markdownify(html2)

    # remove long sequences of '\n'
    mkd1 = re.sub(" {1,}\\n", "\n", mkd)  # remove spaces before \n
    mkd_clean = re.sub("\\n{2,}", "\\n\\n", mkd1)

    return mkd_clean


def aid_to_str(aid):
    """Generate readable string from an aid
    Args :
        aid (dict) : mission transition aid from MT API
    """
    def _repl_none(_str):
        return "" if _str is None else _str

    aid = Dict(aid)
    doc_text = aid.nomAide
    doc_text = doc_text + "\n" + "-" * min(80, len(aid.nomAide)) + "\n"
    doc_text = doc_text + textify_html(aid.description)
    doc_text = doc_text + "\n------------------\n"
    doc_text = doc_text + "id: " + str(aid.id) + "\n"
    doc_text = doc_text + "idSource: " + str(aid.idSource) + "\n"
    doc_text = doc_text + "url: " + _repl_none(aid.urlDescriptif) + "\n"
    doc_text = doc_text + "porteur: " + str(aid.porteursAide) + "\n"
    doc_text = doc_text + "sousThematiques: " + str(
        [th.get('nom', "") for th in aid.sousThematiques]) + "\n"
    doc_text = doc_text + "\n------------------\n"
    doc_text = doc_text + "Exemples de projets:\n\n"
    doc_text = doc_text + textify_html(_repl_none(aid.exempleProjet))

    return doc_text


if __name__ == "__main__":
    mt_aids = Requester().get_aids_mt()

    doccano_json = [{"text": aid_to_str(aid)} for aid in mt_aids]

    with open("doccano_input_ds.json", "w", encoding="utf-8") as fw:
        json.dump(doccano_json, fw, indent=1)
