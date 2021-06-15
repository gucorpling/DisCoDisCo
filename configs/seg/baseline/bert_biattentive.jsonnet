local transformer_model_name = std.extVar("EMBEDDING_MODEL_NAME");
local embedding_dim = std.parseInt(std.extVar("EMBEDDING_DIMS")) + 64 * 2 + 300;
local context_hidden_size = 400;
local encoder_hidden_dim = 256;
local cuda_device = 0;

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
            "token_characters": import "../../components/char_indexer.libsonnet"
        },
        "tokenizer": {
            "type": "whitespace"
        }
    },
    "model": {
        "type": "disrpt_2021_seg_biattentive",
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
                "token_characters": import "../../components/char_embedder.libsonnet"
            }
        },
        // seq2vec encoders for neighbor sentences
        "prev_sentence_encoder": context_encoder,
        "next_sentence_encoder": context_encoder,
        "encoder": {
            "type": "lstm",
            "input_size": encoder_hidden_dim,
            "hidden_size": encoder_hidden_dim,
            "num_layers": 1,
            "bidirectional": true
        },
        "integrator": {
            //"type": "lstm",
            //"input_size": encoder_hidden_dim * 6,
            //"hidden_size": encoder_hidden_dim,
            //"num_layers": 1,
            //"bidirectional": true

            "type": "qanet_encoder",
            "input_dim": encoder_hidden_dim * 3,
            "hidden_dim": encoder_hidden_dim * 2,
            "attention_projection_dim": encoder_hidden_dim,
            "feedforward_hidden_dim": encoder_hidden_dim,
            "num_blocks": 1,
            "num_convs_per_block": 3,
            "conv_kernel_size": 3,
            "num_attention_heads": 8,
            "dropout_prob": 0.2,
            "layer_dropout_undecayed_prob": 0.2,
            "attention_dropout_prob": 0.2,

            //"type": "compose",
            //"encoders": [
            //    {
            //        "type": "pytorch_transformer",
            //        "num_layers": 1,
            //        "num_attention_heads": 8,
            //        "input_dim": encoder_hidden_dim * 2,
            //        "feedforward_hidden_dim": encoder_hidden_dim,
            //    },
            //    {
            //        "type": "feedforward",
            //        "feedforward": {
            //            "input_dim": encoder_hidden_dim * 6,
            //            "num_layers": 2,
            //            "hidden_dims": [encoder_hidden_dim * 4, encoder_hidden_dim * 2],
            //            "activations": "tanh",
            //            "dropout": 0.1
            //        }
            //    }
            //]
        },
        "embedding_dropout": 0.2,
        "encoder_dropout": 0.5,
        "feature_dropout": 0.3,
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
            "lr": 2e-4
        },
        "patience": 7,
        "num_epochs": 60,
	"cuda_device": cuda_device,
	"run_confidence_checks": false,
        // probably best to just use loss
        //"validation_metric": "+span_f1"
    }
}
