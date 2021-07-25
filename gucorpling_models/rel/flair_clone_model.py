import logging
import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.modules.seq2seq_encoders import PytorchTransformer
from allennlp.modules.seq2vec_encoders import LstmSeq2VecEncoder, BertPooler
from overrides import overrides

from allennlp.data import TextFieldTensors, Vocabulary, Token, TokenIndexer
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, FeedForward, Seq2VecEncoder
from allennlp.nn import util, InitializerApplicator, Activation
from allennlp.training.metrics import CategoricalAccuracy
from gucorpling_models.features import Feature, get_feature_modules

logger = logging.getLogger(__name__)


def truncate_sentence(s, max_len=512):
    s["tokens"]["token_ids"] = s["tokens"]["token_ids"][:, :max_len]
    s["tokens"]["mask"] = s["tokens"]["mask"][:, :max_len]
    s["tokens"]["type_ids"] = s["tokens"]["type_ids"][:, :max_len]


@Model.register("disrpt_2021_flair_clone")
class FlairCloneModel(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        dropout: float = 0.4,
        features: Dict[str, Feature] = None,
        initializer: InitializerApplicator = None
    ):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = seq2vec_encoder
        self.dropout = torch.nn.Dropout(dropout)

        # setup handwritten feature modules
        if features is not None and len(features) > 0:
            feature_modules, feature_dims = get_feature_modules(features, vocab)
            self.feature_modules = feature_modules
        else:
            self.feature_modules = None
            feature_dims = 0
        num_relations = vocab.get_vocab_size("relation_labels")

        self.relation_decoder = torch.nn.Linear(
            self.encoder.get_output_dim() + feature_dims + 1, num_relations
        )
        #self.relation_decoder = FeedForward(
        #    input_dim=self.encoder.get_output_dim() + feature_dims + 1,
        #    num_layers=3,
        #    hidden_dims=[512, 256, num_relations],
        #    activations=[Activation.by_name('tanh')(), Activation.by_name('tanh')(), Activation.by_name('linear')()],
        #    dropout=0.1
        #)

        self.relation_accuracy = CategoricalAccuracy()
        self.relation_labels = self.vocab.get_index_to_token_vocabulary("relation_labels")

        if initializer:
            initializer(self)

    def _get_combined_feature_tensor(self, kwargs):
        output_tensors = []
        for module_key, module in self.feature_modules.items():
            output_tensor = module(kwargs[module_key])
            if len(output_tensor.shape) == 1:
                output_tensor = output_tensor.unsqueeze(-1)
            output_tensors.append(output_tensor)

        combined_feature_tensor = torch.cat(output_tensors, dim=1)
        return combined_feature_tensor

    def forward(  # type: ignore
        self,
        combined_body: TextFieldTensors,
        combined_sentence: TextFieldTensors,
        direction: torch.Tensor,
        relation: torch.Tensor = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:

        mask = util.get_text_field_mask(combined_body)
        embedded_sentence = self.embedder(combined_body)
        sentence_embedding = self.encoder(embedded_sentence, mask)

        components = [
            sentence_embedding,
            direction.unsqueeze(-1),
        ]
        if self.feature_modules:
            components.append(self._get_combined_feature_tensor(kwargs))

        combined = torch.cat(components, dim=1)

        # Apply dropout
        combined = self.dropout(combined)

        # Decode the concatenated vector into relation logits
        relation_logits = self.relation_decoder(combined)

        output = {
            "relation_logits": relation_logits,
        }
        if relation is not None:
            self.relation_accuracy(relation_logits, relation)
            output["gold_relation"] = relation
            output["loss"] = F.cross_entropy(relation_logits, relation)
        return output

    # Takes output of forward() and turns tensors into strings or probabilities wherever appropriate
    # Note that the output dict, because it's just from forward(), represents a batch, not a single
    # instance: every key will have a list that's the size of the batch
    def make_output_human_readable(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        # if we have the gold label, decode it into a string
        if "gold_relation" in output_dict:
            output_dict["gold_relation"] = [self.relation_labels[i.item()] for i in output_dict["gold_relation"]]

        # output_dict["relation_logits"] is a tensor of shape (batch_size, num_relations): argmax over the last
        # to get the most likely relation for each instance in the batch
        relation_index = output_dict["relation_logits"].argmax(-1)
        # turn each one into a label
        output_dict["pred_relation"] = [self.relation_labels[i.item()] for i in relation_index]

        # turn relation logits into relation probabilities and present them in a dict
        # where the name of the relation (a string) maps to the probability
        output_dict["relation_probs"] = []
        for relation_logits_row in output_dict["relation_logits"]:
            relation_probs = F.softmax(relation_logits_row)
            output_dict["relation_probs"].append(
                {self.relation_labels[i]: relation_probs[i] for i in range(len(relation_probs))}
            )
        # remove the logits (humans prefer probs)
        del output_dict["relation_logits"]

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "relation_accuracy": self.relation_accuracy.get_metric(reset),  # type: ignore
        }
