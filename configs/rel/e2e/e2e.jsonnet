local transformer_model_name = std.extVar("EMBEDDING_MODEL_NAME");
local max_length = 512;
local feature_size = 10;
local max_span_width = 60;

local transformer_dim = 768;  # uniquely determined by transformer_model

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
        "max_length": 512

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
    "feature_size": feature_size,
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
