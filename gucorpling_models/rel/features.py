import io, os, sys, re
from collections import defaultdict
from argparse import ArgumentParser
from glob import glob


class Token:
    def __init__(self, abs_id, form, xpos, abs_head, deprel, speaker):
        self.id = abs_id
        self.form = form
        self.xpos = xpos
        self.head = abs_head
        self.deprel = deprel
        self.speaker = speaker

    def __repr__(self):
        return self.form + "/"+self.xpos+ "("+str(self.id)+"<-"+self.deprel+"-"+str(self.head)+")"


headers = ["doc","unit1_toks","unit2_toks","unit1_txt","unit2_txt","s1_toks","s2_toks","unit1_sent","unit2_sent","dir","orig_label","label"]

# NLTK stop list, lowercased, plus <*>
nltk_stop = {"<*>", ",", ".", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}


def get_genre(corpus, doc):
    if corpus in ["eng.rst.rstdt","eng.pdtb.pdtb","deu.rst.pcc","por.rst.cstn","zho.pdtb.cdtb"]:
        return "news"
    if corpus == "rus.rst.rrt":
        if "news" in doc:
            return "news"
        else:
            return doc.split("_")[0]
    if corpus == "eng.rst.gum":
        return doc.split("_")[1]
    if corpus == "eng.rst.stac":
        return "chat"
    if corpus == "eus.rst.ert":
        if doc.startswith("SENT"):
            return doc[4:7]
        else:
            return doc[:3]
    if corpus == "fra.sdrt.annodis":
        if doc.startswith("wik"):
            return "wiki"
        else:
            return doc.split("_")[0]
    if corpus in ["nld.rst.nldt","spa.rst.rststb"]:
        return doc[:2]
    if corpus in ["spa.rst.sctb","spa.rst.sctb"]:
        if doc.startswith("TERM"):
            return "TERM"
        else:
            return doc.split("_")[0]
    raise IOError("Unknown corpus: " + corpus)


def get_head_info(unit_span, toks):
    parts = unit_span.split(",")
    covered = []
    for part in parts:
        if "-" in part:
            start, end = part.split("-")
            start = int(start)
            end = int(end)
            covered += list(range(start,end+1))
        else:
            covered.append(int(part))

    head = toks[covered[0]]
    head_dir = "ROOT"
    for i in covered:
        tok = toks[i]
        if tok.head == 0:
            head = tok
            break
        if tok.head < covered[0]:
            head = tok
            head_dir = "LEFT"
            break
        if tok.head > covered[-1]:
            head = tok
            head_dir = "RIGHT"
            break
    return head.deprel, head.xpos, head_dir


def process_relfile(infile, conllu, corpus, as_string=False):
    if not as_string:
        infile = io.open(infile,encoding="utf8").read().strip()
        conllu = io.open(conllu,encoding="utf8").read().strip()

    unit_freqs = defaultdict(int)
    lines = infile.split("\n")
    # Pass 1: Collect number of instances for each head EDU
    for i, line in enumerate(lines):
        if "\t" in line and i > 0:  # Skip header
            cols = {}
            for j, col in enumerate(line.split("\t")):
                cols[headers[j]] = col
            if cols["dir"] == "1>2":
                head_unit = cols["unit2_toks"]
            else:
                head_unit = cols["unit1_toks"]
            unit_freqs[(cols["doc"], head_unit)] += 1

    # Pass 2: Get conllu data
    tokmap = defaultdict(dict)
    toknum = 1
    offset = 0
    speaker = "none"
    docname = None
    fields = [0]
    for line in conllu.split("\n"):
        if "# newdoc" in line:
            docname = line.split("=")[1].strip()
            toknum = 1
            offset = 0
        if "# speaker" in line:
            speaker = line.split("=")[1].strip()
        if "\t" in line:
            fields = line.split("\t")
            if "-" in fields[0] or "." in fields[0]:
                continue
            head = 0 if fields[6] == "0" else int(fields[6]) + offset
            tok = Token(toknum, fields[0], fields[4], head, fields[7], speaker)
            tokmap[docname][toknum] = tok
            toknum += 1
        if len(line) == 0:
            offset += int(fields[0])

    output = []
    # Pass 3: Build features
    for i, line in enumerate(lines):
        if "\t" in line and i > 0:  # Skip header
            feats = {}
            for j, col in enumerate(line.split("\t")):
                feats[headers[j]] = col
            if feats["dir"] == "1>2":
                head_unit = feats["unit2_toks"]
            else:
                head_unit = feats["unit1_toks"]
            feats["nuc_children"] = unit_freqs[(feats["doc"],head_unit)]
            feats["genre"] = get_genre(corpus,feats["doc"])
            feats["u1_discontinuous"] = "<*>" in feats["unit1_txt"]
            feats["u2_discontinuous"] = "<*>" in feats["unit2_txt"]
            feats["u1_issent"] = feats["unit1_txt"] == feats["unit1_sent"]
            feats["u2_issent"] = feats["unit2_txt"] == feats["unit2_sent"]
            feats["u1_length"] = feats["unit1_txt"].replace("<*> ","").count(" ") + 1
            feats["u2_length"] = feats["unit2_txt"].replace("<*> ","").count(" ") + 1
            feats["length_ratio"] = feats["u1_length"]/feats["u2_length"]
            u1_start = re.split(r'[,-]',feats["unit1_toks"])[0]
            u2_start = re.split(r'[,-]',feats["unit2_toks"])[0]
            feats["u1_speaker"] = tokmap[feats["doc"]][int(u1_start)].speaker
            feats["u2_speaker"] = tokmap[feats["doc"]][int(u2_start)].speaker
            feats["same_speaker"] = feats["u1_speaker"] == feats["u2_speaker"]
            u1_func, u1_pos, u1_depdir = get_head_info(feats["unit1_toks"], tokmap[feats["doc"]])
            u2_func, u2_pos, u2_depdir = get_head_info(feats["unit2_toks"], tokmap[feats["doc"]])
            feats["u1_func"] = u1_func
            feats["u1_pos"] = u1_pos
            feats["u1_depdir"] = u1_depdir
            feats["u2_func"] = u2_func
            feats["u2_pos"] = u2_pos
            feats["u2_depdir"] = u2_depdir
            feats["doclen"] = max(tokmap[feats["doc"]])
            feats["u1_position"] = 0.0 if u1_start == "1" else int(u1_start) / feats["doclen"]  # Position as fraction of doc length
            feats["u2_position"] = 0.0 if u2_start == "1" else int(u2_start) / feats["doclen"]  # Position as fraction of doc length
            feats["distance"] = feats["u2_position"] - feats["u1_position"]  # Distance in tokens as fraction of doc length
            unit1_words = feats["unit1_txt"].split(" ")
            unit2_words = feats["unit2_txt"].split(" ")
            overlap_words = [w for w in unit1_words if w in unit2_words and w not in nltk_stop]
            feats["lex_overlap_words"] = " ".join(sorted(overlap_words)) if len(overlap_words) > 0 else "_"
            feats["lex_overlap_length"] = feats["lex_overlap_words"].count(" ") + 1 if len(overlap_words) > 0 else 0
            del feats["unit1_sent"]
            del feats["unit1_toks"]
            del feats["unit1_txt"]
            del feats["s1_toks"]
            del feats["unit2_sent"]
            del feats["unit2_toks"]
            del feats["unit2_txt"]
            del feats["s2_toks"]
            del feats["label"]
            del feats["orig_label"]
            del feats["doc"]
            output.append(feats)

    return output


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--disrpt_data", action="store", default=".."+os.sep+"repo"+os.sep+"data"+os.sep,help="directory with DISRPT repo data folder")
    p.add_argument("-c","--corpus", default="eng.rst.gum", help="corpus name")

    opts = p.parse_args()

    if not opts.disrpt_data.endswith(os.sep):
        opts.disrpt_data += os.sep

    corpus = opts.corpus
    corpus_root = opts.disrpt_data + corpus + os.sep
    files = glob(corpus_root + "*.rels")

    for file_ in files:
        rows = process_relfile(file_, file_.replace(".rels",".conllu"), corpus)

        output = []
        ordered = []
        for row in rows:
            all_keys = row.keys()
            header_keys = [k for k in headers if "label" not in k]
            other_keys = [k for k in row if "label" not in k and k not in headers]
            ordered = header_keys + other_keys + ["label"]
            out_row = []
            for k in ordered:
                if isinstance(row[k],float) and len(str(row[k]))>3:
                    out_row.append("{:.3f}".format(row[k]).rstrip('0'))
                else:
                    out_row.append(str(row[k]))
            output.append("\t".join(out_row))

        output = ["\t".join(ordered)] + output

        with io.open(os.path.basename(file_).replace(".rels","_enriched.rels"),'w',encoding="utf8",newline="\n") as f:
            f.write("\n".join(output)+"\n")