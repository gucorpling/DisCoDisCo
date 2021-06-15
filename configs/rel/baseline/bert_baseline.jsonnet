local transformer_model_name = std.extVar("EMBEDDING_MODEL_NAME");
local embedding_dim = std.parseInt(std.extVar("EMBEDDING_DIMS")) + 64 * 2 + 300;
local encoder_hidden_dim = 256;

local encoder = {
    "type": "lstm",
    "input_size": embedding_dim,
    "hidden_size": encoder_hidden_dim, // 4 <= 2 bilstms applied to 2 sentences
    "num_layers": 1,
    "bidirectional": true,
    "dropout": 0.2
};

// For more info on config files generally, see https://guide.allennlp.org/using-config-files
{
    "dataset_reader" : {
        "type": "disrpt_2021_rel",
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": transformer_model_name
            },
            "fasttext": {
                "type": "single_id",
                "namespace": "fasttext",
            },
            "token_characters": import "../../components/char_indexer.libsonnet"
        },
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": transformer_model_name
        }
    },
    "train_data_path": std.extVar("TRAIN_DATA_PATH"),
    "validation_data_path": std.extVar("VALIDATION_DATA_PATH"),
    "model": {
        "type": "disrpt_2021_rel_baseline",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    // https://docs.allennlp.org/v2.1.0/api/modules/token_embedders/pretrained_transformer_embedder/
                    "type": "pretrained_transformer",
                    "model_name": transformer_model_name,
                    "train_parameters": false,
                    "last_layer_only": false
                },
                "fasttext": {
                    "type": "embedding",
                    "vocab_namespace": "fasttext",
                    "embedding_dim": 300,
                    "trainable": false,
                    "pretrained_file": std.extVar("FASTTEXT_EMBEDDING_FILE")
                },
                "token_characters": import "../../components/char_embedder.libsonnet"
            }
        },
        "encoder1": encoder,
        "encoder2": encoder,
        "dropout": 0.5,
        "feedforward": {
            "input_dim": encoder_hidden_dim * 4 + 1,
            "num_layers": 3,
            "hidden_dims": [512, 256, 128],
            "activations": "relu",
            "dropout": 0.1
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
    },
    "data_loader": {
        "batch_size": 32,
        "shuffle": true
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 3e-4,
            "parameter_groups": [
                [[".*transformer.*"], {"lr": 1e-5}]
            ]
        },
        "num_epochs": 20,
        "cuda_device": 0
    }
}
