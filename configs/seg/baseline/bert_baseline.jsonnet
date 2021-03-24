local transformer_model_name = 'bert-base-cased';
local embedding_dim = 768;
local target_corpus = "eng.rst.gum";

local context_encoder = {
    "type": "lstm",
    "input_size": embedding_dim,
    "hidden_size": 50,
    "num_layers": 1,
    "bidirectional": true,
    "dropout": 0.3
};

// For more info on config files generally, see https://guide.allennlp.org/using-config-files
{
    "dataset_reader" : {
        "type": "disrpt_2021_seg",
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer_mismatched",
                "model_name": transformer_model_name
            }
        },
        "tokenizer": "whitespace"
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
                }
            }
        },
        // seq2vec encoders for neighbor sentences
        "prev_sentence_encoder": context_encoder,
        "next_sentence_encoder": context_encoder,
        // seq2seq encoder for main sentence
        "encoder": {
            "type": "pytorch_transformer", // http://docs.allennlp.org/main/api/modules/seq2seq_encoders/pytorch_transformer_wrapper/
            "input_dim": embedding_dim,
            "num_layers": 3,
            "feedforward_hidden_dim": 128,
            "num_attention_heads": 2,
            "dropout_prob": 0.2
        }
    },
    "train_data_path": "sharedtask2019/data/" + target_corpus + "/" + target_corpus + "_train.conll",
    "validation_data_path": "sharedtask2019/data/" + target_corpus + "/" + target_corpus + "_dev.conll",
    "data_loader": {
        "batch_size": 128,
        "shuffle": true
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.005,
        },
        "num_epochs": 20,
    }
}
