# add categorical features from below to a neural baseline:
# https://github.com/gucorpling/GumDrop2/blob/master/lib/conll_reader.py#L271

import csv
import os
from typing import Dict, Iterable, Any, List, Optional, Tuple
from pprint import pprint

import torch
from allennlp.data import DatasetReader, Instance, Field
from allennlp.data.fields import LabelField, TextField, SequenceLabelField, AdjacencyField, TensorField, ListField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer

from gucorpling_models.seg.gumdrop_reader import read_conll_conn


def group_by_sentence(token_dicts: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    sentences = []

    current_s_id = None
    sentence: List[Dict[str, Any]] = []
    for token in token_dicts:
        s_id = token["s_id"]
        if s_id != current_s_id:
            if sentence:
                sentences.append(sentence)
            sentence = []
            current_s_id = s_id
        sentence.append(token)
    if sentence:
        sentences.append(sentence)
    return sentences


LABEL_TO_ENCODING = {
    "BeginSeg": "B",
    "_": "O",
}


@DatasetReader.register("disrpt_2021_seg")
class Disrpt2021SegReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_tokens: int = None,
        document_boundary_token: str = "@@DOCUMENT_BOUNDARY@@",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens  # useful for BERT
        self.document_boundary_token = document_boundary_token

    def text_to_instance(  # type: ignore
        self,
        sentence: str,
        prev_sentence: Optional[str],
        next_sentence: Optional[str],
        pos_tags: List[str],
        cpos_tags: List[str],
        dep_heads: List[int],
        dep_rels: List[str],
        head_dists: List[str],
        lemmas: List[str],
        morphs: List[str],
        s_type: List[str],
        sent_doc_percentile: List[str],
        labels: List[str],
    ) -> Instance:
        if prev_sentence is None:
            prev_sentence = self.document_boundary_token
        if next_sentence is None:
            next_sentence = self.document_boundary_token
        sentence_tokens = self.tokenizer.tokenize(sentence)
        prev_sentence_tokens = self.tokenizer.tokenize(prev_sentence)
        next_sentence_tokens = self.tokenizer.tokenize(next_sentence)
        if self.max_tokens:
            sentence_tokens = sentence_tokens[: self.max_tokens]
            prev_sentence_tokens = prev_sentence_tokens[: self.max_tokens]
            next_sentence_tokens = next_sentence_tokens[: self.max_tokens]

        if len(sentence_tokens) != len(labels):
            raise ValueError(
                f"Found {len(sentence_tokens)} tokens but {len(labels)} labels. "
                "If you are using a transformer embedding model like BERT, you should be "
                "using a whitespace tokenizer and the special PretrainedTransformerMismatchedIndexer "
                "and PretrainedTransformerMismatchedEmbedder. See: "
                "http://docs.allennlp.org/main/api/data/token_indexers/pretrained_transformer_mismatched_indexer/"
            )

        sentence_field = TextField(sentence_tokens, self.token_indexers)
        # note: if a namespace ends in _tags, it won't get an OOV token automatically. Use only
        # for fields where you're 100% certain all values will occur in train
        fields: Dict[str, Field] = {
            "sentence": sentence_field,
            "prev_sentence": TextField(prev_sentence_tokens, self.token_indexers),
            "next_sentence": TextField(next_sentence_tokens, self.token_indexers),
            "pos_tags": SequenceLabelField(pos_tags, sentence_field, label_namespace="pos_tags"),
            "cpos_tags": SequenceLabelField(cpos_tags, sentence_field, label_namespace="cpos_tags"),
            "dep_heads": TensorField(torch.tensor(dep_heads)),
            "dep_rels": SequenceLabelField(dep_rels, sentence_field, label_namespace="dep_rels_tags"),
            "head_dists": TensorField(torch.tensor(head_dists)),
            "lemmas": SequenceLabelField(lemmas, sentence_field, label_namespace="lemmas"),
            "morphs": SequenceLabelField(morphs, sentence_field, label_namespace="morphs"),
            "s_type": SequenceLabelField(morphs, sentence_field, label_namespace="s_types"),
            "sent_doc_percentile": TensorField(torch.tensor(sent_doc_percentile)),
        }
        if labels:
            fields["labels"] = SequenceLabelField(labels, sentence_field)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        assert file_path.endswith(".conll")

        conll_file_path = file_path
        # tok_file_path = rels_file_path.replace(".conll", ".tok")

        # use gumdrop's function for reading the conll
        token_dicts, _, _, _, _ = read_conll_conn(conll_file_path)
        token_dicts_by_sentence = group_by_sentence(token_dicts)
        sentence_strings = [" ".join(td["word"] for td in sentence) for sentence in token_dicts_by_sentence]

        # syntax -- subtract one from head
        # dep_heads = [
        #     # we need an edge from the parent to this node, zero-indexed. For the root, have it point to itself
        #     # TODO: check how other code handles this.. maybe don't want (i, i)
        #     #[(int(td["head"]) - 1, i) if int(td["head"]) != 0 else (i, i) for i, td in enumerate(sentence)]
        #     for sentence in token_dicts_by_sentence
        # ]

        # syntactic
        pos_tags = [[td["pos"] for td in sentence] for sentence in token_dicts_by_sentence]
        cpos_tags = [[td["cpos"] for td in sentence] for sentence in token_dicts_by_sentence]
        dep_heads = [[int(td["head"]) - 1 for td in sentence] for sentence in token_dicts_by_sentence]
        dep_rels = [[td["deprel"] for td in sentence] for sentence in token_dicts_by_sentence]
        head_dists = [[td["head_dist"] for td in sentence] for sentence in token_dicts_by_sentence]
        lemmas = [[td["lemma"] for td in sentence] for sentence in token_dicts_by_sentence]
        morphs = [[td["morph"] for td in sentence] for sentence in token_dicts_by_sentence]
        s_type = [[td["s_type"] for td in sentence] for sentence in token_dicts_by_sentence]
        sent_doc_percentile = [[td["sent_doc_percentile"] for td in sentence] for sentence in token_dicts_by_sentence]

        # textual

        for i, token_dicts in enumerate(token_dicts_by_sentence):
            prev_sentence = sentence_strings[i - 1] if i > 0 else None
            next_sentence = sentence_strings[i + 1] if i < len(token_dicts_by_sentence) - 1 else None
            sentence = sentence_strings[i]
            labels = [LABEL_TO_ENCODING[td["label"]] for td in token_dicts]
            yield self.text_to_instance(
                sentence=sentence,
                prev_sentence=prev_sentence,
                next_sentence=next_sentence,
                pos_tags=pos_tags[i],
                cpos_tags=cpos_tags[i],
                dep_heads=dep_heads[i],
                dep_rels=dep_rels[i],
                head_dists=head_dists[i],
                lemmas=lemmas[i],
                morphs=morphs[i],
                s_type=s_type[i],
                sent_doc_percentile=sent_doc_percentile[i],
                labels=labels,
            )
