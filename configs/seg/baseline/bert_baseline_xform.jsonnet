local transformer_model_name = std.extVar("EMBEDDING_MODEL_NAME");
local embedding_dim = std.parseInt(std.extVar("EMBEDDING_DIMS")) + 64 * 2 + 300;
local context_hidden_size = 400;
local encoder_hidden_dim = 512;

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
            "fasttext": {
                "type": "single_id",
                "namespace": "fasttext",
            },
            "token_characters": import "../components/char_indexer.libsonnet"
        },
        "tokenizer": {
            "type": "whitespace"
        }
    },
    "model": {
        "type": "disrpt_2021_seg_baseline_xform",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer_mismatched",
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
                "token_characters": import "../components/char_embedder.libsonnet"
            }
        },
        // seq2vec encoders for neighbor sentences
        "prev_sentence_encoder": context_encoder,
        "next_sentence_encoder": context_encoder,
        // our encoder isn't fully configurable here because its input size needs to be determined
        // at the start of the program. so, it'll always be a bilstm, but you can use the two items
        // below to configure the most important hyperparameters it has
        "use_feedforward": false,
        "encoder_hidden_dim": encoder_hidden_dim,
        "encoder_dropout": 0.1,
        // end encoder hyperparams
        "dropout": 0.5,
        "feature_dropout": 0.0
    },
    "train_data_path": std.extVar("TRAIN_DATA_PATH"),
    "validation_data_path": std.extVar("VALIDATION_DATA_PATH"),
    "data_loader": {
        "batch_size": 32,
        "shuffle": true
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 3e-4
        },
        //"learning_rate_scheduler": {
        //    "type": "multi_step",
        //    "milestones": [3],
        //    "gamma": 0.2,
        //},
        "patience": 10,
        "num_epochs": 50,
        // probably best to just use loss
        "validation_metric": "+span_f1"
    }
}
