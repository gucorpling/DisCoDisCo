from typing import Any, List, Dict

import torch
from allennlp.data import Field
from allennlp.data.fields import TensorField, TextField, SequenceLabelField

# Each item should have keys:
# - source_key: key from the token dict returned by the gumdrop code to get the feature data
# - deserialize (optional): 1-arg function to preprocess a value if needed
# - label_namespace (optional): the namespace in the vocab to use to encode label values.
#                               NOTE: OOV tokens will be added unless it ends in _tags
FEATURES: Dict[str, Dict[str, Any]] = {
    # UD pos tags
    "pos_tags": {"source_key": "pos", "label_namespace": "upos"},
    # PTB pos tags
    "cpos_tags": {"source_key": "cpos", "label_namespace": "xpos"},
    # UD syntax
    "dep_heads": {"source_key": "head", "deserialize": lambda val: int(val) - 1},
    "dep_rels": {"source_key": "deprel", "label_namespace": "deprel"},
    # distance (in tokens) to the head
    "head_distances": {"source_key": "head_dist"},
    # length of each token in chars
    "token_lengths": {"source_key": "tok_len"},
    # morphological features (NOTE: might need some breaking up)
    "morphs": {"source_key": "morph", "label_namespace": "morphs"},
    # sentence type (NOTE: might need some breaking up
    "sentence_type": {"source_key": "s_type", "label_namespace": "s_types"},
    # how deep into the doc we are TODO: verify
    "document_depth": {"source_key": "sent_doc_percentile"},
    # length of the sentence
    "sentence_length": {"source_key": "s_len"},
    # whether the token is a bracket TODO: verify
    "token_is_bracket": {"source_key": "bracket"},
}


def get_field(key, data: List[Any], sentence: TextField) -> Field:
    """
    Used by a reader's text_to_instance method to convert from data to AllenNLP Fields
    """
    if key not in FEATURES:
        raise Exception(f"Unknown categorical feature: {key}")
    config = FEATURES[key]
    py_type = type(data[0])

    if py_type in [int, float]:
        return TensorField(torch.tensor(data))
    elif py_type == str:
        return SequenceLabelField(data, sentence, label_namespace=config.get("label_namespace", "labels"))
    else:
        raise Exception(f"Unsupported type for categorical feature: {py_type}")


# NYI:
# genre

# textual (support NYI)
# lemma
# heading_first
# heading_last
# first
# last

# Not sure what it is:
# depchunk
# case
# conj
# quote
# wid
# parentclauses
