"""Extract ids and labels from doccano dataset export"""
import re
import json
import argparse


def extract_aid_infos(aid_text):
    """extract ids and title from a doccano item"""
    # extract ids
    reg_ids = r"id: (\d+)\nidSource: (at_(\d+))"
    mob = re.search(reg_ids, aid_text)
    id_mt, id_at = int(mob.group(1)), mob.group(2)
    title = aid_text.splitlines()[0]

    return id_mt, id_at, title


def main_parser(ds_annotated):
    """Parse extracted json from doccano"""
    ret = []
    for aid in ds_annotated:
        id_mt, id_at, title = extract_aid_infos(aid.get("text"))
        ret.append({"id_mt": id_mt, "id_at": id_at, "title": title,
                    "labels": aid.get("label")})
    return ret


if __name__ == "__main__":
    """ main """
    parser = argparse.ArgumentParser(description="Extract ids and labels from"
                                     " doccano dataset export")
    parser.add_argument('--inp', type=str, help='input file (doccano export)')
    parser.add_argument('--out', type=str, default="doccano_parsed.json",
                        help='ouput file (ids + labels)')

    args = parser.parse_args()

    with open(args.inp, encoding="utf-8") as fin:
        ds_annotated = json.load(fin)

    json_out = main_parser(ds_annotated)

    with open(args.out, 'w', encoding="utf-8") as fout:
        json.dump(json_out, fout)
