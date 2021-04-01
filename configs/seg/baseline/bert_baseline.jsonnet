local transformer_model_name = std.extVar("EMBEDDING_MODEL_NAME");
local embedding_dim = std.parseInt(std.extVar("EMBEDDING_DIMS")) + 64 * 2;
local context_hidden_size = 400;
local encoder_hidden_dim = 256;

local context_encoder = {
    "type": "lstm",
    "input_size": embedding_dim,
    "hidden_size": context_hidden_size / 4, // 4 <= 2 bilstms applied to 2 sentences
    "num_layers": 1,
    "bidirectional": true,
    "dropout": 0.2
};

// For more info on config files generally, see https://guide.allennlp.org/using-config-files
{
    "dataset_reader" : {
        "type": "disrpt_2021_seg",
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer_mismatched",
                "model_name": transformer_model_name
            },
            "token_characters": import "../components/char_indexer.libsonnet"
        },
        "tokenizer": {
            "type": "whitespace"
        }
    },
    "model": {
        "type": "disrpt_2021_seg_baseline",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer_mismatched",
                    "model_name": transformer_model_name,
                    "train_parameters": false,
                    "last_layer_only": false
                },
                "token_characters": import "../components/char_embedder.libsonnet"
            }
        },
        // seq2vec encoders for neighbor sentences
        "prev_sentence_encoder": context_encoder,
        "next_sentence_encoder": context_encoder,
        // our encoder isn't fully configurable here because its input size needs to be determined
        // at the start of the program. so, it'll always be a bilstm, but you can use the two items
        // below to configure the most important hyperparameters it has
        "encoder_hidden_dim": encoder_hidden_dim,
        "encoder_recurrent_dropout": 0.3,
        // end encoder hyperparams
        "dropout": 0.5,
        "feature_dropout": 0.4
    },
    "train_data_path": std.extVar("TRAIN_DATA_PATH"),
    "validation_data_path": std.extVar("VALIDATION_DATA_PATH"),
    "data_loader": {
        "batch_size": 16,
        "shuffle": true
    },
    "trainer": {
        "optimizer": {
            "type": "adamw",
            "lr": 5e-4
        },
        "patience": 6,
        "num_epochs": 30,
        // probably best to just use loss
        //"validation_metric": "+span_f1"
    }
}
