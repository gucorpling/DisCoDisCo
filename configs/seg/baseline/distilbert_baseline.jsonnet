local transformer_model_name = 'distilbert-base-cased';
local embedding_dim = 768 + 40 * 2;
local context_hidden_size = 200;
local sentence_encoder_dim = embedding_dim + 9;
local encoder_input_dim = sentence_encoder_dim * 2 + context_hidden_size * 4;
local target_corpus = "eng.rst.gum";

local context_encoder = {
    "type": "lstm",
    "input_size": embedding_dim,
    "hidden_size": context_hidden_size,
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
        // seq2seq encoder for main sentence
        "sentence_encoder": {
            "type": "stacked_bidirectional_lstm",
            "num_layers": 1,
            "input_size": sentence_encoder_dim,
            "hidden_size": sentence_encoder_dim,
            "recurrent_dropout_probability": 0.3,
            "layer_dropout_probability": 0.3,
        },
        // seq2seq encoder for the final output to the decoder
        "encoder": {
            "type": "stacked_bidirectional_lstm",
            "num_layers": 1,
            "input_size": encoder_input_dim,
            "hidden_size": 512,
            "recurrent_dropout_probability": 0.3,
            "layer_dropout_probability": 0.3,
        },
        "dropout": 0.5
    },

    "train_data_path": "sharedtask2019/data/" + target_corpus + "/" + target_corpus + "_train.conll",
    "validation_data_path": "sharedtask2019/data/" + target_corpus + "/" + target_corpus + "_dev.conll",
    "data_loader": {
        "batch_size": 64,
        "shuffle": true
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 20
    }
}
