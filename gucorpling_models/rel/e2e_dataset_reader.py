import csv
import io, os
from typing import Dict, Iterable, Optional, List, Tuple
from pprint import pprint

from overrides import overrides

from allennlp.data import DatasetReader, Instance, Field
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer
from gucorpling_models.rel.e2e_util import make_coref_instance
# from allennlp_models.coref.util import make_coref_instance
from gucorpling_models.seg.gumdrop_reader import read_conll_conn
from gucorpling_models.seg.dataset_reader import group_by_sentence


def mapping(tok_doc: List[str], conll_doc: List[str]) -> Dict[str, Dict[int, Tuple[int, int]]]:
    """
    Mapping token indices and sentence information
    """
    docs = {}
    map_dict = {}
    # count_tok = 1
    count_conll = 0
    sent_id = 0
    doc_name = ""
    for idx, line in enumerate(tok_doc):
        if idx == len(tok_doc) - 1:
            docs[doc_name] = map_dict
        if line.startswith("# newdoc id"):
            new_doc_name = line.split("=")[-1].strip()
            if map_dict:
                docs[doc_name] = map_dict
                map_dict = {}
            doc_name = new_doc_name
            count_tok = 1
        elif "\t" in line:
            tok_id = count_tok
            while '\t' not in conll_doc[count_conll]:
                if conll_doc[count_conll].startswith("# sent_id"):
                    sent_id = int(conll_doc[count_conll].split("-")[-1]) - 1
                count_conll += 1

            conll_id = int(conll_doc[count_conll].split("\t")[0]) - 1
            map_dict[tok_id] = (sent_id, conll_id)
            count_conll += 1
            count_tok += 1
    return docs


def get_sents(conll_doc: List[str]) -> Dict[str, List[List[str]]]:
    """
    Get sentences for each doc
    """
    doc_of_sents = {}
    sents = []
    sent = []
    doc_name = ""
    for idx, line in enumerate(conll_doc):
        if idx == len(conll_doc) - 1:
            doc_of_sents[doc_name] = sents
        elif line:
            if line.startswith("# newdoc id"):
                new_doc_name = line.split("=")[-1].strip()
                if sents:
                    sents.append(sent)
                    if doc_name:
                        doc_of_sents[doc_name] = sents
                    sents = []
                    sent = []
                doc_name = new_doc_name

            if line.startswith("# sent_id"):
                if sent:
                    sents.append(sent)
                    sent = []
            elif line.startswith("#"):
                continue
            else:
                sent.append(line.split("\t")[1])
    return doc_of_sents


def merge_spans(doc, tok_conll_mapping, doc_of_sents, left_end, right_start, right_end):
    # merge right to left to avoid the gap

    left_end_tok_id = tok_conll_mapping[doc][left_end]
    right_start_tok_id = tok_conll_mapping[doc][right_start]
    right_end_tok_id = tok_conll_mapping[doc][right_end]
    gap_words = doc_of_sents[doc][right_start_tok_id[0]][left_end_tok_id[1] + 1:right_start_tok_id[1]]
    right_words = doc_of_sents[doc][right_start_tok_id[0]][right_start_tok_id[1]:right_end_tok_id[1] + 1]

    assert left_end_tok_id[0] == right_start_tok_id[0]

    return doc_of_sents[doc][right_start_tok_id[0]][:left_end_tok_id[1] + 1] \
            + right_words \
            + gap_words \
            + doc_of_sents[doc][right_start_tok_id[0]][right_end_tok_id[1] + 1:]


@DatasetReader.register("disrpt_2021_rel_e2e")
class Disrpt2021RelReader(DatasetReader):
    def __init__(
        self,
        max_span_width: int,
        token_indexers: Dict[str, TokenIndexer] = None,
        wordpiece_modeling_tokenizer: Optional[PretrainedTransformerTokenizer] = None,
        max_sentences: int = None,
        remove_singleton_clusters: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._wordpiece_modeling_tokenizer = wordpiece_modeling_tokenizer
        self._max_sentences = max_sentences
        self._remove_singleton_clusters = remove_singleton_clusters

    @overrides
    def text_to_instance(
        self,  # type: ignore
        sentences: List[List[str]],
        gold_clusters: Optional[List[List[Tuple[int, int]]]],
        dir: List,
        label: List
    ) -> Instance:
        return make_coref_instance(
            sentences,
            self._token_indexers,
            self._max_span_width,
            gold_clusters,
            self._wordpiece_modeling_tokenizer,
            self._max_sentences,
            self._remove_singleton_clusters,
            dir,
            label
        )

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        Tentatively we regard EDUs are in one cluster in a document. It should be noted that this is only a baseline
          solution. One possible solution is to treat each level of a RST tree as a cluster, e.g. Level 2 may have
          x individual subtrees so that it has x clusters.
        It is also important to know the difference between this task and the coreference resolution task that different
          spans can point back to the same antecedent in the relation classification task. There needs to be a better
          solution to handle this issue.

        Return:
            sentences: `List[List[str]]`
            gold_clusters: `List[List[Tuple[int, itn]]]`
                Each element is a tuple composed of (cluster_id, (start_index, end_index)).
            dir: `List[Tuple[str, Tuple[Span1, Span2]]]`, Span=Tuple[int, int]
            label: `List[Tuple[str, Tuple[Span1, Span2]]]`
        """
        assert file_path.endswith(".rels")

        rels_file_path = file_path
        conll_file_path = rels_file_path.replace(".rels", ".conllu")
        tok_file_path = rels_file_path.replace(".rels", ".tok")

        tok_conll_mapping = mapping(
            io.open(tok_file_path, encoding='utf-8').read().split('\n'),
            io.open(conll_file_path, encoding='utf-8').read().split('\n')
        )
        doc_of_sents = get_sents(io.open(conll_file_path, encoding='utf-8').read().split('\n'))

        non_seen_doc = []
        seen = {}
        docs = {}
        with io.open(rels_file_path, "r", encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                doc = row["doc"]
                unit1_toks = row["unit1_toks"]
                unit1_txt = row["unit1_txt"]
                unit1_sent = row["unit1_sent"]
                unit2_toks = row["unit2_toks"]
                unit2_txt = row["unit2_txt"]
                unit2_sent = row["unit2_sent"]
                dir = row["dir"]
                label = row["label"]

                if doc not in doc_of_sents.keys():
                    non_seen_doc.append(doc)
                    continue
                if doc not in docs.keys():
                    docs[doc] = {"spans": [], "dir": [], "label": []}
                if doc not in seen.keys():
                    seen[doc] = []
                if "," in unit1_toks:
                    left1, right1 = unit1_toks.split(",")[0], unit1_toks.split(",")[1]
                    left1_start, left1_end = int(left1.split('-')[0]), int(left1.split('-')[-1])
                    right1_start, right1_end = int(right1.split('-')[0]), int(right1.split('-')[-1])

                    if unit1_toks not in seen[doc]:
                        sent_id1 = tok_conll_mapping[doc][right1_start][0]
                        doc_of_sents[doc][sent_id1] = merge_spans(doc, tok_conll_mapping, doc_of_sents, left1_end,
                                                                  right1_start, right1_end)
                        seen[doc].append(unit1_toks)
                    span = (left1_start, right1_end - (right1_start-left1_end+1))
                else:
                    span = (int(unit1_toks.split('-')[0]), int(unit1_toks.split('-')[-1]))

                if "," in unit2_toks:
                    left2, right2 = unit2_toks.split(",")[0], unit2_toks.split(",")[1]
                    left2_start, left2_end = int(left2.split('-')[0]), int(left2.split('-')[-1])
                    right2_start, right2_end = int(right2.split('-')[0]), int(right2.split('-')[-1])
                    if unit2_toks not in seen[doc]:
                        sent_id2 = tok_conll_mapping[doc][right2_start][0]
                        doc_of_sents[doc][sent_id2] = merge_spans(doc, tok_conll_mapping, doc_of_sents, left2_end,
                                                                  right2_start, right2_end)
                        seen[doc].append(unit2_toks)
                    next_span = (left2_start, right2_end - (right2_start - left2_end + 1))
                else:
                    next_span = (int(unit2_toks.split('-')[0]), int(unit2_toks.split('-')[-1]))

                if span not in docs[doc]["spans"]:
                    docs[doc]["spans"].append(span)
                docs[doc]["dir"].append((dir, (span, next_span)))
                docs[doc]["label"].append((label, (span, next_span)))

        for docname, doc_dict in docs.items():
            yield self.text_to_instance(
                sentences=doc_of_sents[docname],
                gold_clusters=[docs[docname]["spans"]],
                dir=docs[docname]["dir"],
                label=docs[docname]["label"]
            )
