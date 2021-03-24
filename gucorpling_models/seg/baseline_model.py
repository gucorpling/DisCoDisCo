from typing import Dict, Optional, Any

import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules import (
    TextFieldEmbedder,
    Seq2VecEncoder,
    Seq2SeqEncoder,
    TimeDistributed,
    FeedForward,
    ConditionalRandomField,
)
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.nn import util, Activation
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure


@Model.register("disrpt_2021_seg_baseline")
class Disrpt2021Baseline(Model):
    """
    A simple encoder-decoder baseline which embeds all four spans (each unit's sentence and discourse unit),
    uses a seq2vec encoder to encode each span, and decodes using a simple linear transform
    for the direction and the relation.

    Based in part on https://github.com/allenai/allennlp-models/blob/main/allennlp_models/tagging/models/crf_tagger.py
    """

    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        prev_sentence_encoder: Seq2VecEncoder,
        next_sentence_encoder: Seq2VecEncoder,
        dropout: float = 0.4,
    ):
        super().__init__(vocab)
        print(vocab)
        print(vocab.get_index_to_token_vocabulary("labels"))

        # encoding --------------------------------------------------
        self.embedder = embedder
        self.encoder = encoder  # todo: use a transformer wrapper
        self.prev_sentence_encoder = prev_sentence_encoder  # todo: use an lstm
        self.next_sentence_encoder = next_sentence_encoder

        num_labels = vocab.get_vocab_size("labels")
        self.dropout = torch.nn.Dropout(dropout)

        hidden_size = (
            encoder.get_output_dim() + prev_sentence_encoder.get_output_dim() + next_sentence_encoder.get_output_dim()
        )

        # decoding --------------------------------------------------
        self.tag_projection_layer = TimeDistributed(torch.nn.Linear(hidden_size, num_labels))

        # TODO: implement contrained decoding
        self.crf = ConditionalRandomField(num_labels, None, include_start_end_transitions=True)

        # util --------------------------------------------------
        # these are stateful objects that keep track of accuracy across an epoch
        self.label_accuracy = CategoricalAccuracy()
        self._label_f1 = F1Measure(1)

        # convenience dict mapping indices to labels
        self.labels = self.vocab.get_index_to_token_vocabulary("labels")

    def forward(  # type: ignore
        self,
        sentence: TextFieldTensors,
        prev_sentence: TextFieldTensors,
        next_sentence: TextFieldTensors,
        labels: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)

        # Encoding --------------------------------------------------
        # Embed the text. Shape: (batch_size, num_tokens, embedding_dim)
        embedded_sentence = self.embedder(sentence)
        embedded_prev_sentence = self.embedder(prev_sentence)
        embedded_next_sentence = self.embedder(next_sentence)

        # Encode the context. Shape: (batch_size, encoder_output_dim)
        encoded_prev_sentence = self.prev_sentence_encoder(embedded_prev_sentence, get_text_field_mask(prev_sentence))
        encoded_next_sentence = self.next_sentence_encoder(embedded_next_sentence, get_text_field_mask(next_sentence))
        context = torch.cat((encoded_prev_sentence, encoded_next_sentence), 1)

        # Encode the sentence into (batch_size, num_tokens, h_0)
        encoded_sentence = self.encoder(embedded_sentence, mask)
        # Context is of shape (batch_size, h_1 + h_2).
        # This turns it into (batch_size, num_tokens, h_1 + h_2)
        time_distributed_context = context.unsqueeze(1).expand(-1, encoded_sentence.shape[1], -1)
        # We now have (batch_size, num_tokens, h_0 + h_1 + h_2)
        encoded_sequence = torch.cat((encoded_sentence, time_distributed_context), dim=2)

        encoded_sequence = self.dropout(encoded_sequence)

        # Decoding --------------------------------------------------
        label_logits = self.tag_projection_layer(encoded_sequence)
        best_label_seqs = self.crf.viterbi_tags(label_logits, mask, top_k=None)
        # each in the batch gets a (tags, viterbi_score) pair
        # just take the tags, ignore the viterbi score
        pred_labels = [best_label_seq for best_label_seq, _ in best_label_seqs]

        output = {
            "label_logits": label_logits,
            "mask": mask,
            "pred_labels": pred_labels,
        }
        if labels is not None:
            output["gold_labels"] = labels

            # Add negative log-likelihood as loss
            log_likelihood = self.crf(label_logits, labels, mask)
            output["loss"] = -log_likelihood

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the metrics
            class_probabilities = label_logits * 0.0
            for i, instance_labels in enumerate(pred_labels):
                for j, label_id in enumerate(instance_labels):
                    class_probabilities[i, j, label_id] = 1

            self.label_accuracy(class_probabilities, labels, mask)
            self._label_f1(class_probabilities, labels, mask)

        return output

    # Takes output of forward() and turns tensors into strings or probabilities wherever appropriate
    # Note that the output dict, because it's just from forward(), represents a batch, not a single
    # instance: every key will have a list that's the size of the batch
    def make_output_human_readable(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        def decode_labels(labels):
            return [self.vocab.get_token_from_index(int(label), "labels") for label in labels]

        output_dict["pred_labels"] = [decode_labels(t) for t in output_dict["pred_labels"]]
        output_dict["gold_labels"] = [decode_labels(t) for t in output_dict["gold_labels"]]

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "accuracy": self.label_accuracy.get_metric(reset),  # type: ignore
            "_f1": self._label_f1.get_metric(reset),  # type: ignore
        }
