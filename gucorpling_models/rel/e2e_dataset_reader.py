import csv
import io, os
import math
from typing import Dict, Iterable, Optional, List, Tuple
from pprint import pprint
from collections import defaultdict

from overrides import overrides

import torch
from torch import tensor
from allennlp.data import DatasetReader, Instance, Field
from allennlp.data.fields import LabelField, TextField, TensorField, ArrayField, ListField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerMismatchedIndexer
from allennlp.data.tokenizers import Token, Tokenizer, PretrainedTransformerTokenizer
from gucorpling_models.seg.gumdrop_reader import read_conll_conn
from gucorpling_models.seg.dataset_reader import group_by_sentence


def get_span_indices(unit_toks, s_toks, max_length: None):
    s_start, s_end = int(s_toks.split("-")[0]), int(s_toks.split("-")[-1])

    if "," in unit_toks:
        left, right = unit_toks.split(",")[0], unit_toks.split(",")[1]
        left_start, left_end = int(left.split("-")[0]) - s_start, int(left.split("-")[-1]) - s_start
        right_start, right_end = int(right.split("-")[0])-s_start, int(right.split("-")[-1])-s_start
        if max_length:
            if max_length >= left_end:
                left_end = max_length - 1
                span = [(left_start, left_end)]
            elif max_length >= right_start:
                span = [(left_start, left_end)]
            elif max_length >= right_end:
                right_end = max_length - 1
                span = [(left_start, left_end), (right_start, right_end)]
        else:
            span = [(left_start, left_end), (right_start, right_end)]
    else:
        left, right = int(unit_toks.split("-")[0])-s_start, int(unit_toks.split("-")[-1])-s_start
        if max_length and right >= max_length:
            right = max_length - 1
        span = [(left, right)]
    return span


def get_span_dist(unit1, unit2):
    unit1_start_indice = int(unit1.split(",")[0].split("-")[0])
    unit2_start_indice = int(unit2.split(",")[0].split("-")[0])
    return unit1_start_indice-unit2_start_indice




@DatasetReader.register("disrpt_2021_rel_e2e")
class Disrpt2021RelReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_length: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_length = max_length  # useful for BERT

    def tokenize_with_subtoken_map(self, text, span):
        subtoken_map = [0]
        token_to_subtokens = {}
        count = 1

        tokenized_text = [Token('[CLS]')]
        for i, word in enumerate(text.split(' ')):
            tokenized = self.tokenizer.tokenize(word)
            if self.max_length and i >= self.max_length:
                break
            tokenized_text += tokenized[1:-1]
            subtoken_map += [i+1] * (len(tokenized)-2)
            token_to_subtokens[i] = (count, count+len(tokenized)-3)
            count += len(tokenized)-2
        tokenized_text.append(Token('[SEP]'))
        subtoken_map.append(len(subtoken_map))

        span_mask = [0] * len(tokenized_text)
        for i, s in enumerate(span):
            s_start, s_end = s
            new_start, new_end = token_to_subtokens[s_start][0], token_to_subtokens[s_end][-1]
            span[i] = (new_start, new_end)
            for x in range(new_start, new_end+1):
                span_mask[x] = 1

        assert len(subtoken_map) == len(tokenized_text)
        return tokenized_text, span, span_mask

    @overrides
    def text_to_instance(
        self,  # type: ignore
        # unit1_txt: str,
        unit1_sent: str,
        # unit2_txt: str,
        unit2_sent: str,
        span_dist: int,
        unit1_span_indices: list,
        unit2_span_indices: list,
        dir: str,
        label: str = None,
    ) -> Instance:

        unit1_sent_tokens, new_unit1_span_indices, unit1_span_mask = self.tokenize_with_subtoken_map(unit1_sent, unit1_span_indices)
        unit2_sent_tokens, new_unit2_span_indices, unit2_span_mask = self.tokenize_with_subtoken_map(unit2_sent, unit2_span_indices)

        sent_tokens = unit1_sent_tokens + unit2_sent_tokens[1:]
        adjusted_unit1_span_mask = unit1_span_mask + [0] * (len(unit2_span_mask)-1)
        adjusted_unit2_span_mask = [0] * len(unit1_span_mask) + unit2_span_mask[1:]

        # unit1_txt_tokens = self.tokenizer.tokenize(unit1_txt)
        # unit2_txt_tokens = self.tokenizer.tokenize(unit2_txt)

        fields: Dict[str, Field] = {
            "sentences": TextField(sent_tokens, self.token_indexers),
            "unit1_span_mask": TensorField(torch.tensor(adjusted_unit1_span_mask) > 0),
            "unit2_span_mask": TensorField(torch.tensor(adjusted_unit2_span_mask) > 0),
            "direction": LabelField(dir, label_namespace="direction_labels"),
            "distance": TensorField(torch.tensor(span_dist))
        }
        if label:
            fields["relation"] = LabelField(label, label_namespace="relation_labels")
        return Instance(fields)


    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        assert file_path.endswith(".rels")

        with io.open(file_path, "r", encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                # doc = row["doc"]
                unit1_toks = row["unit1_toks"]
                s1_toks = row["s1_toks"]
                unit2_toks = row["unit2_toks"]
                s2_toks = row["s2_toks"]

                span_dist = get_span_dist(unit1_toks, unit2_toks)
                span_dist = abs(span_dist) // 10 + 1         # smooth the distance
                unit1_span_indices = get_span_indices(unit1_toks, s1_toks, self.max_length)
                unit2_span_indices = get_span_indices(unit2_toks, s2_toks, self.max_length)

                yield self.text_to_instance(
                    unit1_sent=row["unit1_sent"],
                    unit2_sent=row["unit2_sent"],
                    span_dist=span_dist,
                    unit1_span_indices=unit1_span_indices,
                    unit2_span_indices=unit2_span_indices,
                    dir=row["dir"],
                    label=row["label"],
                )
