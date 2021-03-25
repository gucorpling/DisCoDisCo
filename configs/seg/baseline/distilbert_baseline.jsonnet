local transformer_model_name = 'distilbert-base-cased';
local embedding_dim = 768 + 80 * 2;
local context_hidden_size = 200;
local sentence_encoder_dim = embedding_dim;
local feature_count = 11;
// sentence_encoder is a bilstm, so multiply by 2. context is gotten with a bilstm on both sides, so multilply by 4
local encoder_input_dim = sentence_encoder_dim * 2 + context_hidden_size * 4 + feature_count;
local encoder_hidden_dim = 512;
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
            "layer_dropout_probability": 0.1,
        },
        // seq2seq encoder for the final output to the decoder
        "encoder1": {
            "type": "stacked_bidirectional_lstm",
            "num_layers": 1,
            "input_size": encoder_input_dim,
            "hidden_size": encoder_hidden_dim,
            "recurrent_dropout_probability": 0.3,
            "layer_dropout_probability": 0.1,
        },
        "encoder2": {
            "type": "pass_through",
            "input_dim": encoder_hidden_dim * 2
        },
        "dropout": 0.5,
        "feature_dropout": 0.5,
        "proportion_loss_without_out_tag": 0.0
    },
    "train_data_path": "sharedtask2019/data/" + target_corpus + "/" + target_corpus + "_train.conll",
    "validation_data_path": "sharedtask2019/data/" + target_corpus + "/" + target_corpus + "_dev.conll",
    "data_loader": {
        "batch_size": 64,
        "shuffle": true
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 5e-4
        },
        "patience": 5,
        "num_epochs": 30,
        //"validation_metric": "+f1"
    }
}
