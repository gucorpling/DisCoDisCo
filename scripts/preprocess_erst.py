# Assumes relevant data at `data/rstpp` and `data/gum`
import os
from collections import defaultdict

import conllu
from lxml import etree

CACHE = {}


def parse_xml(file_path):
    if file_path in CACHE:
        return CACHE[file_path]
    else:
        tree = etree.parse(file_path)
        root = tree.getroot()
        CACHE[file_path] = root
        return root


def tokens_of_doc(xml):
    body = xml.find("body")
    text_tokens = " ".join(n.text for n in body.findall("segment"))
    return text_tokens.split(" ")


def parse_range(s):
    output = []
    for p in s.split(","):
        if "-" not in p:
            output.append((int(p) - 1, int(p)))
        else:
            x, y = p.split("-")
            output.append((int(x) - 1, int(y)))
    return output


def tokens_of_range(all_tokens, r):
    out = []
    parsed = parse_range(r)
    for x, y in parsed:
        out.append(all_tokens[x:y])
    return out


def tokens_to_text(tokens):
    return " <*> ".join(" ".join(s) for s in tokens)


def dm_locations(xml):
    tokens = [
        sorted([int(i) - 1 for i in n.attrib["tokens"].split(",")])
        for n in xml.findall(".//signal")
        if n.attrib["type"] == "dm" and n.attrib["subtype"] == "dm"
    ]
    return tokens


def add_dm_tokens(row, all_tokens, dm_locs, left="{", right="}"):
    new_row = row.copy()

    def inner(unit):
        ranges = parse_range(row[f"unit{unit}_toks"])
        tokens_unit = tokens_of_range(all_tokens, row[f"unit{unit}_toks"])
        for i, range_ in enumerate(map(lambda x: range(*x), ranges)):
            for loc in dm_locs:
                if all(x in range_ for x in loc):
                    offset = range_[0]
                    if list(range(loc[0], loc[-1] + 1)) == loc:
                        tokens_unit[i][loc[0] - offset] = "=LEFT=" + tokens_unit[i][loc[0] - offset]
                        tokens_unit[i][loc[-1] - offset] = tokens_unit[i][loc[-1] - offset] + "=RIGHT="
                    else:
                        for j in loc:
                            tokens_unit[i][j - offset] = "=LEFT=" + tokens_unit[i][j - offset] + "=RIGHT="
        new_row[f"unit{unit}_txt"] = (
            tokens_to_text(tokens_unit).replace("=LEFT=", left).replace("=RIGHT=", right)
        )
        #if row[f"unit{unit}_txt"] != new_row[f"unit{unit}_txt"]:
        #    print()
        #    print(row[f"unit{unit}_txt"])
        #    print(new_row[f"unit{unit}_txt"])
        #    input()

    inner(1)
    inner(2)
    return new_row


def process_row(row):
    xml = parse_xml(f"data/rstpp/data/results/rs4/{row['doc']}.rs4")
    all_tokens = tokens_of_doc(xml)

    tokens_unit1 = tokens_of_range(all_tokens, row["unit1_toks"])
    tokens_unit2 = tokens_of_range(all_tokens, row["unit2_toks"])

    # Ensure reconstructed matches original
    reconstructed_unit1 = tokens_to_text(tokens_unit1)
    reconstructed_unit2 = tokens_to_text(tokens_unit2)
    # print()
    # print(f"data/rstpp/data/results/rs4/{row['doc']}.rs4")
    # print(row["unit1_txt"])
    # print(reconstructed_unit1 )
    # print(row["unit2_txt"])
    # print(reconstructed_unit2 )
    assert reconstructed_unit1 == row["unit1_txt"]
    assert reconstructed_unit2 == row["unit2_txt"]

    dm_locs = dm_locations(xml)
    new_row = add_dm_tokens(row, all_tokens, dm_locs)
    return new_row


def process_split(condition, split):
    fieldnames = "doc unit1_toks unit2_toks unit1_txt unit2_txt s1_toks s2_toks unit1_sent unit2_sent dir orig_label label".split(" ")
    with open(f"data/gum/rst/disrpt/eng.rst.gum_{split}.rels", 'r') as f:
        s = f.read().strip()
        rows = [{k: v for k, v in zip(fieldnames, r.split("\t"))} for r in s.split("\n")[1:]]

    new_rows = []
    for row in rows:
        new_rows.append(process_row(row))

    with open(f"data/erst/{condition}/eng.rst.gum_{split}.rels", 'w') as f:
        f.write("\t".join(fieldnames) + "\n")
        for r in new_rows:
            vals = []
            for n in fieldnames:
                vals.append(r[n])
            f.write("\t".join(vals) + "\n")


def process_row_silver(doc_info, row):
    all_tokens = doc_info["tokens"]

    tokens_unit1 = tokens_of_range(all_tokens, row["unit1_toks"])
    tokens_unit2 = tokens_of_range(all_tokens, row["unit2_toks"])

    # Ensure reconstructed matches original
    reconstructed_unit1 = tokens_to_text(tokens_unit1)
    reconstructed_unit2 = tokens_to_text(tokens_unit2)

    assert reconstructed_unit1 == row["unit1_txt"]
    assert reconstructed_unit2 == row["unit2_txt"]

    dm_locs = doc_info["dm_locs"]
    new_row = add_dm_tokens(row, all_tokens, dm_locs)
    return new_row


def transfer_seg(treebanked, pred):
    def include_token(t):
        return isinstance(t["id"], int) or "." in t["id"]
    assert len(treebanked) == len(pred)
    for s1, s2 in zip(treebanked, pred):
        ts1 = [t for t in s1 if include_token(t)]
        ts2 = [t for t in s2 if include_token(t)]
        assert len(ts1) == len(ts2)
        assert all(ts1[i]["form"] == ts2[i]["form"] for i in range(len(ts1)))
        for t1, t2 in zip(ts1, ts2):
            if "misc" in t2 and t2["misc"] is not None and "Seg" in t2["misc"]:
                if "misc" not in t1 or t1["misc"] is None:
                    t1["misc"] = {}
                t1["misc"]["Seg"] = t2["misc"]["Seg"]


def get_doc_info(treebanked):
    doc_info = defaultdict(lambda: defaultdict(list))

    def include_token(t):
        return isinstance(t["id"], int)
    for s in treebanked:
        doc_name = s.metadata["sent_id"].split("-")[0]
        conn = None
        for t in s:
            if include_token(t):
                if "misc" in t and t["misc"] is not None and "Seg" in t["misc"]:
                    tag = t["misc"]["Seg"][0]
                    if tag == "B":
                        conn = [len(doc_info[doc_name]["tokens"])]
                    elif tag == "I":
                        conn.append(len(doc_info[doc_name]["tokens"]))
                    else:
                        doc_info[doc_name]["dm_locs"].append(conn)
                        conn = None
                elif conn is not None:
                    doc_info[doc_name]["dm_locs"].append(conn)
                    conn = None
                doc_info[doc_name]["tokens"].append(t["form"])
        if conn is not None:
            doc_info[doc_name]["dm_locs"].append(conn)
    return doc_info


def process_split_silver(condition, split):
    fieldnames = "doc unit1_toks unit2_toks unit1_txt unit2_txt s1_toks s2_toks unit1_sent unit2_sent dir orig_label label".split(" ")
    with open(f"data/gum/rst/disrpt/eng.rst.gum_{split}.rels", 'r') as f:
        s = f.read().strip()
        rows = [{k: v for k, v in zip(fieldnames, r.split("\t"))} for r in s.split("\n")[1:]]

    with open(f"data/2021/eng.pdtb.gum/eng.pdtb.gum_{split}.conllu", 'r') as f:
        treebanked_sentences = conllu.parse(f.read())
        # Delete gold
        for s in treebanked_sentences:
            for t in s:
                if "misc" in t and t["misc"] is not None and "Seg" in t["misc"]:
                    del t["misc"]["Seg"]
    with open(f"models/eng.pdtb.gum_seg_bert_baseline_ft/output_test.conllu", "r") as f:
        pred_sentences = conllu.parse(f.read())

    transfer_seg(treebanked_sentences, pred_sentences)
    doc_info = get_doc_info(treebanked_sentences)

    new_rows = []
    for row in rows:
        new_rows.append(process_row_silver(doc_info[row["doc"]], row))

    with open(f"data/erst/{condition}_silver/eng.rst.gum_{split}.rels", 'w') as f:
        f.write("\t".join(fieldnames) + "\n")
        for r in new_rows:
            vals = []
            for n in fieldnames:
                vals.append(r[n])
            f.write("\t".join(vals) + "\n")


def combine_conllu(condition, split):
    dev_docs = ["GUM_interview_cyclone", "GUM_interview_gaming",
                "GUM_news_iodine", "GUM_news_homeopathic",
                "GUM_voyage_athens", "GUM_voyage_coron",
                "GUM_whow_joke", "GUM_whow_overalls",
                "GUM_bio_byron", "GUM_bio_emperor",
                "GUM_fiction_lunre", "GUM_fiction_beast",
                "GUM_academic_exposure", "GUM_academic_librarians",
                "GUM_reddit_macroeconomics", "GUM_reddit_pandas",  # Reddit
                "GUM_speech_impeachment", "GUM_textbook_labor",
                "GUM_vlog_radiology", "GUM_conversation_grounded",
                "GUM_textbook_governments", "GUM_vlog_portland",
                "GUM_conversation_risk", "GUM_speech_inauguration"]
    test_docs = ["GUM_interview_libertarian", "GUM_interview_hill",
                 "GUM_news_nasa", "GUM_news_sensitive",
                 "GUM_voyage_oakland", "GUM_voyage_vavau",
                 "GUM_whow_mice", "GUM_whow_cactus",
                 "GUM_fiction_falling", "GUM_fiction_teeth",
                 "GUM_bio_jespersen", "GUM_bio_dvorak",
                 "GUM_academic_eegimaa", "GUM_academic_discrimination",
                 "GUM_reddit_escape", "GUM_reddit_monsters",  # Reddit
                 "GUM_speech_austria", "GUM_textbook_chemistry",
                 "GUM_vlog_studying", "GUM_conversation_retirement",
                 "GUM_textbook_union", "GUM_vlog_london",
                 "GUM_conversation_lambada", "GUM_speech_newzealand"]
    if split == "dev":
        docs = dev_docs
    elif split == "test":
        docs = test_docs
    else:
        docs = os.listdir("data/gum/dep")
        docs = [d[:-7] for d in docs]
        docs = [d for d in docs if d not in dev_docs and d not in test_docs]

    with open(f"data/erst/{condition}/eng.rst.gum_{split}.conllu", "w") as f:
        for doc in docs:
            with open(f"data/gum/dep/{doc}.conllu", "r") as f2:
                f.write(f2.read())


def process_splits():
    condition = "curly"
    for split in ["train", "dev", "test"]:
        process_split(condition, split)
        combine_conllu(condition, split)
    process_split_silver(condition, "test")


if __name__ == '__main__':
    process_splits()