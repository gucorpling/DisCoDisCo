# add categorical features from below to a neural baseline:
# https://github.com/gucorpling/GumDrop2/blob/master/lib/conll_reader.py#L271

import csv
import os
from typing import Dict, Iterable, Any, List, Optional
from pprint import pprint

from allennlp.data import DatasetReader, Instance, Field
from allennlp.data.fields import LabelField, TextField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer

from gucorpling_models.seg.util import read_conll_conn


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
        fields: Dict[str, Field] = {
            "sentence": sentence_field,
            "prev_sentence": TextField(prev_sentence_tokens, self.token_indexers),
            "next_sentence": TextField(next_sentence_tokens, self.token_indexers),
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

        for i, token_dicts in enumerate(token_dicts_by_sentence):
            prev_sentence = sentence_strings[i - 1] if i > 0 else None
            next_sentence = sentence_strings[i + 1] if i < len(token_dicts_by_sentence) - 1 else None
            sentence = sentence_strings[i]
            yield self.text_to_instance(
                sentence=sentence,
                prev_sentence=prev_sentence,
                next_sentence=next_sentence,
                labels=[td["label"] for td in token_dicts],
            )