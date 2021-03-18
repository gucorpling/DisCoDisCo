from typing import Dict, Optional, Any

import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, FeedForward
from allennlp.nn import util
from allennlp.nn.util import get_device_of
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("disrpt_2021_baseline")
class Disrpt2021Baseline(Model):
    """
    A simple encoder-decoder baseline which embeds all four spans (each unit's sentence and discourse unit),
    uses a seq2vec encoder to encode each span, and decodes using a simple linear transform
    for the direction and the relation.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        encoder1: Seq2VecEncoder,
        encoder2: Seq2VecEncoder,
        encoder_decoder_dropout: float = 0.4,
    ):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder1 = encoder1
        self.encoder2 = encoder2

        num_directions = vocab.get_vocab_size("direction_labels")
        num_relations = vocab.get_vocab_size("relation_labels")
        self.dropout = torch.nn.Dropout(encoder_decoder_dropout)

        linear_input_size = encoder1.get_output_dim() * 2 + encoder2.get_output_dim() * 2
        self.direction_decoder = torch.nn.Linear(linear_input_size, num_directions)
        self.relation_decoder = torch.nn.Linear(linear_input_size, num_relations)

        # these are stateful objects that keep track of accuracy across an epoch
        self.relation_accuracy = CategoricalAccuracy()
        self.direction_accuracy = CategoricalAccuracy()

    def forward(  # type: ignore
        self,
        unit1_body: TextFieldTensors,
        unit1_sentence: TextFieldTensors,
        unit2_body: TextFieldTensors,
        unit2_sentence: TextFieldTensors,
        direction: torch.Tensor = None,
        relation: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:

        # Embed the text. Shape: (batch_size, num_tokens, embedding_dim)
        embedded_unit1_body = self.embedder(unit1_body)
        embedded_unit1_sentence = self.embedder(unit1_sentence)
        embedded_unit2_body = self.embedder(unit2_body)
        embedded_unit2_sentence = self.embedder(unit2_sentence)

        # Encode the text. Shape: (batch_size, encoder_output_dim)
        encoded_unit1_body = self.encoder1(embedded_unit1_body, util.get_text_field_mask(unit1_body))
        encoded_unit1_sentence = self.encoder1(embedded_unit1_sentence, util.get_text_field_mask(unit1_sentence))
        encoded_unit2_body = self.encoder2(embedded_unit2_body, util.get_text_field_mask(unit2_body))
        encoded_unit2_sentence = self.encoder2(embedded_unit2_sentence, util.get_text_field_mask(unit2_sentence))

        # Concatenate the vectors. Shape: (batch_size, encoder1_output_dim * 2 + encoder2_output_dim * 2)
        combined = torch.cat(
            (
                encoded_unit1_body,
                encoded_unit1_sentence,
                encoded_unit2_body,
                encoded_unit2_sentence,
            ),
            1,
        ).to(get_device_of(encoded_unit1_body))
        combined = self.dropout(combined)

        direction_logits = self.direction_decoder(combined)
        relation_logits = self.relation_decoder(combined)

        output = {
            "direction_logits": direction_logits,
            "relation_logits": relation_logits,
        }
        if direction is not None and relation is not None:
            self.direction_accuracy(direction_logits, direction)
            self.relation_accuracy(relation_logits, relation)
            output["gold_direction"] = direction
            output["gold_relation"] = relation

            relation_loss = F.cross_entropy(relation_logits, relation)
            direction_loss = F.cross_entropy(direction_logits, direction)
            output["loss"] = relation_loss + direction_loss
        return output

    def make_output_human_readable(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        if "gold_direction" in output_dict:
            output_dict["gold_direction"] = [
                self.vocab.get_token_from_index(int(i), "direction_labels") for i in output_dict["gold_direction"]
            ]
        if "gold_relation" in output_dict:
            output_dict["gold_relation"] = [
                self.vocab.get_token_from_index(int(i), "relation_labels") for i in output_dict["gold_relation"]
            ]

        direction_index = output_dict["direction_logits"].argmax(-1)
        output_dict["pred_direction"] = [
            self.vocab.get_token_from_index(int(i), "direction_labels") for i in direction_index
        ]
        relation_index = output_dict["relation_logits"].argmax(-1)
        output_dict["pred_relation"] = [
            self.vocab.get_token_from_index(int(i), "relation_labels") for i in relation_index
        ]

        output_dict["direction_probs"] = list()
        for direction_logits_row in output_dict["direction_logits"]:
            direction_probs = F.softmax(direction_logits_row)
            output_dict["direction_probs"].append(
                {
                    self.vocab.get_token_from_index(int(i), "direction_labels"): direction_probs[i]
                    for i in range(len(direction_probs))
                }
            )
        del output_dict["direction_logits"]

        output_dict["relation_probs"] = []
        for relation_logits_row in output_dict["relation_logits"]:
            relation_probs = F.softmax(relation_logits_row)
            output_dict["relation_probs"].append(
                {
                    self.vocab.get_token_from_index(int(i), "relation_labels"): relation_probs[i]
                    for i in range(len(relation_probs))
                }
            )
        del output_dict["relation_logits"]

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "direction_accuracy": self.direction_accuracy.get_metric(reset),  # type: ignore
            "relation_accuracy": self.relation_accuracy.get_metric(reset),  # type: ignore
        }
