import logging
import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules import FeedForward, GatedSum
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.nn import util, InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy

logger = logging.getLogger(__name__)


@Model.register("disrpt_2021_e2e")
class E2eResolver(Model):

    def __init__(
            self,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            context_layer: Seq2SeqEncoder,
            mention_feedforward: FeedForward,
            antecedent_feedforward: FeedForward,
            feature_size: int,
            max_span_width: int,
            spans_per_word: float,
            max_antecedents: int,
            coarse_to_fine: bool = False,
            inference_order: int = 1,
            lexical_dropout: float = 0.2,
            encoder_decoder_dropout: float = 0.3,
            initializer: InitializerApplicator = InitializerApplicator(),
            **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)

        self._text_field_embedder = text_field_embedder
        self._context_layer = context_layer
        self._global_attention = TimeDistributed(torch.nn.Linear(text_field_embedder.get_output_dim(), 1))

        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x
        self._distance_embedding = torch.nn.Embedding(num_embeddings=150, embedding_dim=feature_size)
        self._dir_embedding = torch.nn.Embedding(num_embeddings=2, embedding_dim=1)

        num_relations = vocab.get_vocab_size("relation_labels")
        self.dropout = torch.nn.Dropout(encoder_decoder_dropout)
        self.relation_decoder = torch.nn.Linear(context_layer.get_output_dim()*2+feature_size+1, num_relations)

        # these are stateful objects that keep track of accuracy across an epoch
        self.direction_accuracy = CategoricalAccuracy()
        self.relation_accuracy = CategoricalAccuracy()

        # convenience dict mapping relation indices to labels
        self.relation_labels = self.vocab.get_index_to_token_vocabulary("relation_labels")
        initializer(self)

    def _get_span_embeddings(self, unit_sentence, unit_span_mask, unit_sentence_mask):
        embedded_unit_sentence = self._lexical_dropout(self._text_field_embedder(unit_sentence))
        assert unit_span_mask.size(1) == embedded_unit_sentence.size(1)
        contextualized_unit_sentence = self._context_layer(embedded_unit_sentence, unit_sentence_mask)

        att_logits = self._global_attention(contextualized_unit_sentence)    # [b, s, 1]
        att_logits[~unit_span_mask] = float('-inf')
        att_weights = F.softmax(att_logits, 1)  # [b, s, 1]
        att_span_embeds = torch.mul(contextualized_unit_sentence, att_weights.expand_as(contextualized_unit_sentence))   # [b, s, e]
        weighted_span_embeds = att_span_embeds.sum(dim=1)   # [b, e]
        return weighted_span_embeds

    @overrides
    def forward(
            self,  # type: ignore
            sentences: TextFieldTensors,
            unit1_span_mask: torch.Tensor,
            unit2_span_mask: torch.Tensor,
            direction: torch.Tensor,
            distance: torch.Tensor,
            relation: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:

        sentences_mask = util.get_text_field_mask(sentences)
        unit1_span_embeddings = self._get_span_embeddings(sentences, unit1_span_mask, sentences_mask)   # [b, e]
        unit2_span_embeddings = self._get_span_embeddings(sentences, unit2_span_mask, sentences_mask)   # [b, e]

        dist_embeds = self._distance_embedding(distance)
        dir_embeds = self._dir_embedding(direction)
        feature_embeds = torch.cat((dist_embeds, dir_embeds), -1) # [b, f]

        combined = torch.cat((unit1_span_embeddings, unit2_span_embeddings, feature_embeds), 1) # [b, e*2+f]
        self.dropout(combined)
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


    @overrides
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

    def _compute_span_pair_embeddings(
            self,
            top_span_embeddings: torch.FloatTensor,
            antecedent_embeddings: torch.FloatTensor,
            antecedent_offsets: torch.FloatTensor,
    ):
        """
        Computes an embedding representation of pairs of spans for the pairwise scoring function
        to consider. This includes both the original span representations, the element-wise
        similarity of the span representations, and an embedding representation of the distance
        between the two spans.

        # Parameters

        top_span_embeddings : `torch.FloatTensor`, required.
            Embedding representations of the top spans. Has shape
            (batch_size, num_spans_to_keep, embedding_size).
        antecedent_embeddings : `torch.FloatTensor`, required.
            Embedding representations of the antecedent spans we are considering
            for each top span. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, embedding_size).
        antecedent_offsets : `torch.IntTensor`, required.
            The offsets between each top span and its antecedent spans in terms
            of spans we are considering. Has shape (batch_size, num_spans_to_keep, max_antecedents).

        # Returns

        span_pair_embeddings : `torch.FloatTensor`
            Embedding representation of the pair of spans to consider. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        target_embeddings = top_span_embeddings.unsqueeze(2).expand_as(antecedent_embeddings)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        antecedent_distance_embeddings = self._distance_embedding(
            util.bucket_values(antecedent_offsets, num_total_buckets=self._num_distance_buckets)
        )

        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        span_pair_embeddings = torch.cat(
            [
                target_embeddings,
                antecedent_embeddings,
                antecedent_embeddings * target_embeddings,
                antecedent_distance_embeddings,
            ],
            -1,
        )
        return span_pair_embeddings