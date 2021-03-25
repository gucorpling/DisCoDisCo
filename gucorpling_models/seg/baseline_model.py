from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules import (
    TextFieldEmbedder,
    Seq2VecEncoder,
    Seq2SeqEncoder,
    TimeDistributed,
    ConditionalRandomField,
)
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.nn import util, Activation, InitializerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, F1Measure, SpanBasedF1Measure

from gucorpling_models.seg.util import detect_encoding
from gucorpling_models.seg.features import FEATURES


def get_handcrafted_feature_tensor(kwargs):
    return torch.cat([kwargs[key].unsqueeze(-1) for key in FEATURES.keys()], dim=2)


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
        sentence_encoder: Seq2SeqEncoder,
        prev_sentence_encoder: Seq2VecEncoder,
        next_sentence_encoder: Seq2VecEncoder,
        initializer: InitializerApplicator = InitializerApplicator(),
        dropout: float = 0.5,
        feature_dropout: float = 0.3,
        proportion_loss_without_out_tag: float = 0.0,
    ):
        super().__init__(vocab)
        print(vocab)
        for k, v in vocab._index_to_token.items():
            if len(v) < 50:
                print(k)
                print(" ", v)
        self.labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(self.labels)
        assert 0.0 <= proportion_loss_without_out_tag <= 1.0, "Proportion must be between 0 and 1"
        self.proportion_loss_without_out_tag = proportion_loss_without_out_tag

        # encoding --------------------------------------------------
        self.embedder = embedder
        self.encoder = encoder
        self.sentence_encoder = sentence_encoder
        self.prev_sentence_encoder = prev_sentence_encoder
        self.next_sentence_encoder = next_sentence_encoder
        self.dropout = torch.nn.Dropout(dropout)
        self.feature_dropout = torch.nn.Dropout(feature_dropout)

        hidden_size = encoder.get_output_dim()

        # decoding --------------------------------------------------
        self.label_projection_layer = TimeDistributed(torch.nn.Linear(hidden_size, num_labels))

        # encoding scheme
        label_encoding, encoding_map = detect_encoding(self.labels)
        self._encoding_map = encoding_map
        constraints = allowed_transitions(label_encoding, self.labels)
        self.crf = ConditionalRandomField(num_labels, constraints, include_start_end_transitions=True)

        # util --------------------------------------------------
        # these are stateful objects that keep track of accuracy across an epoch
        self.metrics = {"span_f1": SpanBasedF1Measure(vocab, tag_namespace="labels", label_encoding=label_encoding)}

        initializer(self)
        print(self)

    def _get_encoded_context(self, prev_sentence, next_sentence, num_tokens):
        """
        Represent neighboring sentences using seq2vec encoders. The shape of the vec is
        (batch_size, context_encoder_hidden_dim * 2), but we expand this to
        (batch_size, num_toks, context_encoder_hidden_dim * 2) in order to make it easy to combine with the sequence
        """
        embedded_prev_sentence = self.embedder(prev_sentence)
        embedded_next_sentence = self.embedder(next_sentence)
        encoded_prev_sentence = self.prev_sentence_encoder(embedded_prev_sentence, get_text_field_mask(prev_sentence))
        encoded_next_sentence = self.next_sentence_encoder(embedded_next_sentence, get_text_field_mask(next_sentence))
        context = torch.cat((encoded_prev_sentence, encoded_next_sentence), 1)
        # Context is of shape (batch_size, h_1 + h_2).
        # This turns it into (batch_size, num_tokens, h_1 + h_2)
        time_distributed_context = context.unsqueeze(1).expand(-1, num_tokens, -1)
        return time_distributed_context

    def forward(  # type: ignore
        self,
        sentence: TextFieldTensors,
        prev_sentence: TextFieldTensors,
        next_sentence: TextFieldTensors,
        original_sentence_length: List[int],
        labels: torch.LongTensor = None,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)

        # Encoding --------------------------------------------------
        embedded_sentence = self.embedder(sentence)
        encoded_sentence = self.sentence_encoder(embedded_sentence, mask)

        # Get our handcrafted features as well as our context from neighboring sentences, and put it all together
        # into the input for the final encoder
        combined_feature_tensor = get_handcrafted_feature_tensor(kwargs)
        combined_feature_tensor = self.feature_dropout(combined_feature_tensor)
        context = self._get_encoded_context(prev_sentence, next_sentence, embedded_sentence.shape[1])
        encoder_input = torch.cat((encoded_sentence, context, combined_feature_tensor), dim=2)

        encoded_sequence = self.encoder(encoder_input, mask)
        encoded_sequence = self.dropout(encoded_sequence)

        # Decoding --------------------------------------------------
        # project into the label space and use viterbi decoding on the CRF
        label_logits = self.label_projection_layer(encoded_sequence)
        pred_labels = [
            best_label_seq for best_label_seq, viterbi_score in self.crf.viterbi_tags(label_logits, mask, top_k=None)
        ]

        output = {
            "label_logits": label_logits,
            "mask": mask,
            "pred_labels": pred_labels,
        }
        if labels is not None:
            self._compute_loss_and_metrics(label_logits, pred_labels, labels, mask, output)

        return output

    def _compute_loss_and_metrics(self, label_logits, pred_labels, labels, mask, output):
        output["gold_labels"] = labels

        # Add negative log-likelihood as loss
        if self.proportion_loss_without_out_tag:
            o_tag_index = self._encoding_map["O"]
            no_o_mask = mask & (labels != o_tag_index)
            no_o_log_likelihood = self.crf(label_logits, labels, no_o_mask)
            log_likelihood = self.crf(label_logits, labels, mask)

            output["loss"] = -(
                self.proportion_loss_without_out_tag * no_o_log_likelihood
                + (1 - self.proportion_loss_without_out_tag) * log_likelihood
            )
        else:
            log_likelihood = self.crf(label_logits, labels, mask)
            output["loss"] = -log_likelihood

        # Represent viterbi labels as "class probabilities" that we can feed into the metrics
        class_probabilities = label_logits * 0.0
        for i, instance_labels in enumerate(pred_labels):
            for j, label_id in enumerate(instance_labels):
                class_probabilities[i, j, label_id] = 1

        for metric in self.metrics.values():
            metric(class_probabilities, labels, mask)

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
            k.replace("-overall", "").replace("-measure", ""): v
            for k, v in self.metrics["span_f1"].get_metric(reset=reset).items()
            if "overall" in k
        }
