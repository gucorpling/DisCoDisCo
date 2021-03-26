local transformer_model_name = 'distilbert-multilingual-base-cased';
local embedding_dim = 768 + 64 * 2;
local context_hidden_size = 200;
local encoder_hidden_dim = 256;
local target_corpus = "zho.rst.sctb";

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
                    "train_parameters": true,
                    //"last_layer_only": false
                },
                "token_characters": import "../components/char_embedder.libsonnet"
            }
        },
        // seq2vec encoders for neighbor sentences
        "prev_sentence_encoder": context_encoder,
        "next_sentence_encoder": context_encoder,
        "encoder_hidden_dim": encoder_hidden_dim,
        "encoder_recurrent_dropout": 0.3,
        "dropout": 0.5,
        "feature_dropout": 0.3
    },
    "train_data_path": "sharedtask2019/data/" + target_corpus + "/" + target_corpus + "_train.conll",
    "validation_data_path": "sharedtask2019/data/" + target_corpus + "/" + target_corpus + "_dev.conll",
    "data_loader": {
        "batch_size": 8,
        "shuffle": true
    },
    "trainer": {
        "optimizer": {
            "type": "adamw",
            "lr": 3e-3
        },
        "patience": 5,
        "num_epochs": 30,
        "validation_metric": "+span_f1"
    }
}