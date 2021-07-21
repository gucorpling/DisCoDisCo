import math
from typing import Tuple, Dict, List, Any, Union

import torch
from allennlp.common import FromParams, Registrable
import scipy.stats as stats


from allennlp.data import Vocabulary, Field
from allennlp.data.fields import TextField, TensorField, SequenceLabelField, LabelField


class TransformationFunction(Registrable):
    def __call__(self, xs, tokenwise):
        raise NotImplementedError("Please use a class that inherits from TransformationFunction")


@TransformationFunction.register("natural_log")
class NaturalLog(TransformationFunction):
    def __call__(self, xs, tokenwise):
        if tokenwise:
            return [math.log(x) if x != 0 else x for x in xs]
        else:
            return math.log(xs)


@TransformationFunction.register("abs_natural_log")
class NaturalLog(TransformationFunction):
    def __call__(self, xs, tokenwise):
        if tokenwise:
            return [math.log(abs(x)) if x != 0 else x for x in xs]
        else:
            return math.log(abs(xs))


class Feature(FromParams):
    def __init__(self, source_key: str, label_namespace: str = None, xform_fn: TransformationFunction = None):
        self.source_key = source_key
        self.label_namespace = label_namespace
        self.xform_fn = xform_fn


def get_feature_field(feature_config: Feature, features: Union[List[Any], Any], sentence: TextField = None) -> Field:
    """
    Returns an AllenNLP `Field` suitable for use on an AllenNLP `Instance` for a given token-level feature.
    If the type of the data in `features` is int or float, we will use TensorField; if it is str, we will
    use SequenceLabelField; other data types are currently unsupported.

    Args:
        feature_config: a Feature for the Field
        features: either a list of data (token-wise features) or a single piece of data
        sentence: the TextField the Field is associated with. If present, some fields will be associated with it.

    Returns:
        a Field for the feature.
    """
    tokenwise = (isinstance(features, list) or isinstance(features, tuple))
    if tokenwise and not (len(features) == len(sentence.tokens)):
        raise Exception(f"Token-level features must match the number of tokens")

    if feature_config.xform_fn is not None:
        features = feature_config.xform_fn(features, tokenwise=tokenwise)

    py_type = type(features[0]) if tokenwise else type(features)
    if py_type in [int, float]:
        return TensorField(torch.tensor(features))
    elif py_type == str and tokenwise:
        return SequenceLabelField(features, sentence, label_namespace=feature_config.label_namespace or "labels")
    elif py_type == str:
        return LabelField(features, label_namespace=feature_config.label_namespace or "labels")
    elif py_type == bool and not tokenwise:
        return LabelField("true" if features else "false", label_namespace=feature_config.label_namespace or "labels")
    else:
        raise Exception(f"Unsupported type for feature: {py_type}")


def get_feature_modules(
    features: Dict[str, Feature], vocab: Vocabulary
) -> Tuple[torch.nn.ModuleDict, int]:
    """
    Returns a PyTorch `ModuleDict` containing a module for each feature in `token_features`.
    This function tries to be smart: if the feature is numeric, it will not do anything, but
    if it is categorical (as indicated by the presence of a `label_namespace`), then the module
    will be a `torch.nn.Embedding` with size equal to the ceiling of the square root of the
    categorical feature's vocabulary size. We could be a lot smarter of course, but this will
    get us going.

    Args:
        features: a dict of `TokenFeatures` describing all the categorical features to be used
        vocab: the initialized vocabulary for the model

    Returns:
        A 2-tuple: the ModuleDict, and the summed output dimensions of every module, for convenience.
    """
    modules: Dict[str, torch.nn.Module] = {}
    total_dims = 0
    for key, config in features.items():
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
