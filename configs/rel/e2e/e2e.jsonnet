local transformer_model_name = std.extVar("EMBEDDING_MODEL_NAME");
local max_length = 512;
local feature_size = 10;
local max_span_width = 60;

local transformer_dim = 768;  # uniquely determined by transformer_model

local features = {
    "nuc_children": {"source_key": "nuc_children"},
    "genre": {"source_key": "genre", "label_namespace": "genre"},
    "u1_discontinuous": {"source_key": "u1_discontinuous", "label_namespace": "discontinuous"},
    "u2_discontinuous": {"source_key": "u2_discontinuous", "label_namespace": "discontinuous"},
    "u1_issent": {"source_key": "u1_issent", "label_namespace": "issent"},
    "u2_issent": {"source_key": "u2_issent", "label_namespace": "issent"},
    "u1_length": {"source_key": "u1_length"},
    "u2_length": {"source_key": "u2_length"},
    "length_ratio": {"source_key": "length_ratio"},
    "u1_speaker": {"source_key": "u1_speaker", "label_namespace": "speaker"},
    "u2_speaker": {"source_key": "u2_speaker", "label_namespace": "speaker"},
    "same_speaker": {"source_key": "same_speaker", "label_namespace": "same_speaker"},
    "u1_func": {"source_key": "u1_func", "label_namespace": "func"},
    "u1_pos": {"source_key": "u1_pos", "label_namespace": "pos"},
    "u1_depdir": {"source_key": "u1_depdir", "label_namespace": "depdir"},
    "u2_func": {"source_key": "u2_func", "label_namespace": "func"},
    "u2_pos": {"source_key": "u2_pos", "label_namespace": "pos"},
    "u2_depdir": {"source_key": "u2_depdir", "label_namespace": "depdir"},
    "doclen": {"source_key": "doclen"},
    "u1_position": {"source_key": "u1_position"},
    "u2_position": {"source_key": "u2_position"},
    "distance": {"source_key": "distance"},
    //"lex_overlap_words": {"source_key": "lex_overlap_words", "label_namespace": "lex_overlap_words"},
    "lex_overlap_length": {"source_key": "lex_overlap_length"}
};

{
    "dataset_reader" : {
        "type": "disrpt_2021_rel_e2e",
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": transformer_model_name
            }
        },
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": transformer_model_name
        },
        "max_length": 512,
        "features": features
    },
  "train_data_path": std.extVar("TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("VALIDATION_DATA_PATH"),
  "model": {
    "type": "disrpt_2021_e2e",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "pretrained_transformer",
            "model_name": transformer_model_name,
            "train_parameters": false,
            "last_layer_only": false
        }
      }
    },
    "context_layer": {
        "type": "pass_through",
        "input_dim": transformer_dim
    },
    "initializer": {
      "regexes": [
        [".*_span_updating_gated_sum.*weight", {"type": "xavier_normal"}],
        [".*linear_layers.*weight", {"type": "xavier_normal"}],
        [".*scorer.*weight", {"type": "xavier_normal"}],
        ["_distance_embedding.weight", {"type": "xavier_normal"}],
        ["_span_width_embedding.weight", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_ih.*", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_hh.*", {"type": "orthogonal"}]
      ]
    },
    # "feature_size": feature_size,
    "features": features,
  },
    "data_loader": {
        "batch_size": 32,
        "shuffle": true
    },
  "trainer": {
    "num_epochs": 40,
    "patience" : 10,
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 3e-4,
      "parameter_groups": [
        [[".*transformer.*"], {"lr": 1e-5}]
      ]
    }
  }
}
