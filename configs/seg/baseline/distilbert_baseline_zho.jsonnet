local transformer_model_name = 'distilbert-base-multilingual-cased';
local embedding_dim = 768;
local target_corpus = "zho.rst.sctb";

local context_encoder = {
    "type": "lstm",
    "input_size": embedding_dim,
    "hidden_size": 100,
    "num_layers": 1,
    "bidirectional": true
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
            "num_layers": 1,
            "feedforward_hidden_dim": 512,
            "num_attention_heads": 4,
        }
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
