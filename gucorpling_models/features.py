import math
from typing import Tuple, Dict, List, Any

import torch
from allennlp.common import Params, FromParams, Registrable
import json

from allennlp.data import Vocabulary, Field
from allennlp.data.fields import TextField, TensorField, SequenceLabelField


class DeserializeFunction(Registrable):
    def __call__(self, x):
        raise NotImplementedError("Please use a class that inherits from DeserializeFunction")


@DeserializeFunction.register("int")
class Int(DeserializeFunction):
    def __call__(self, x):
        return int(x)


@DeserializeFunction.register("float")
class Float(DeserializeFunction):
    def __call__(self, x):
        return float(x)


class TokenFeature(FromParams):
    def __init__(self, source_key: str, label_namespace: str = None, deserialize_fn: DeserializeFunction = None):
        self.source_key = source_key
        self.label_namespace = label_namespace
        self.deserialize_fn = deserialize_fn


def get_token_feature_field(token_feature: TokenFeature, features: List[Any], sentence: TextField) -> Field:
    """
    Returns an AllenNLP `Field` suitable for use on an AllenNLP `Instance` for a given token-level feature.
    If the type of the data in `features` is int or float, we will use TensorField; if it is str, we will
    use SequenceLabelField; other data types are currently unsupported.

    Args:
        token_feature: a TokenFeature for the Field
        features: the token-level features that we are creating a Field for
        sentence: the TextField the Field is associated with--needed for

    Returns:
        a Field for the feature.
    """
    if not (len(features) == len(sentence.tokens)):
        raise Exception(f"Token-level features must match the number of tokens")

    py_type = type(features[0])
    if py_type in [int, float]:
        return TensorField(torch.tensor(features))
    elif py_type == str:
        return SequenceLabelField(features, sentence, label_namespace=token_feature.label_namespace or "labels")
    else:
        raise Exception(f"Unsupported type for feature: {py_type}")


def get_token_feature_modules(
    token_features: Dict[str, TokenFeature], vocab: Vocabulary
) -> Tuple[torch.nn.ModuleDict, int]:
    """
    Returns a PyTorch `ModuleDict` containing a module for each feature in `token_features`.
    This function tries to be smart: if the feature is numeric, it will not do anything, but
    if it is categorical (as indicated by the presence of a `label_namespace`), then the module
    will be a `torch.nn.Embedding` with size equal to the ceiling of the square root of the
    categorical feature's vocabulary size. We could be a lot smarter of course, but this will
    get us going.

    Args:
        token_features: a dict of `TokenFeatures` describing all the categorical features to be used
        vocab: the initialized vocabulary for the model

    Returns:
        A 2-tuple: the ModuleDict, and the summed output dimensions of every module, for convenience.
    """
    modules: Dict[str, torch.nn.Module] = {}
    total_dims = 0
    for key, config in token_features.items():
        ns = config.label_namespace
        if ns is None:
            modules[key] = torch.nn.Identity()
            total_dims += 1
        else:
            size = vocab.get_vocab_size(ns)
            # if size <= 5:
            #    modules[key] = torch.nn.Identity()
            #    total_dims += size
            # else:
            edims = math.ceil(math.sqrt(size))
            total_dims += edims
            modules[key] = torch.nn.Embedding(size, edims, padding_idx=(0 if vocab.is_padded(ns) else None))

    return torch.nn.ModuleDict(modules), total_dims
