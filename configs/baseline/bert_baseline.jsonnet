local transformer_model_name = 'bert-base-cased';

// For more info on config files generally, see https://guide.allennlp.org/using-config-files
{
    "dataset_reader" : {
        "type": "disrpt_2021",
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": transformer_model_name
            }
        },
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": transformer_model_name
        }
    },
    "train_data_path": "sharedtask2021/data/eng.rst.gum/eng.rst.gum_dev.rels",
    "validation_data_path": "sharedtask2021/data/eng.rst.gum/eng.rst.gum_dev.rels",
    "model": {
        "type": "disrpt_2021_baseline",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    // https://docs.allennlp.org/v2.1.0/api/modules/token_embedders/pretrained_transformer_embedder/
                    "type": "pretrained_transformer",
                    "model_name": transformer_model_name
                }
            }
        },
        "encoder1": {
            "type": "bert_pooler",
            "pretrained_model": transformer_model_name,
            "dropout": 0.3
        },
        "encoder2": {
            "type": "bert_pooler",
            "pretrained_model": transformer_model_name,
            "dropout": 0.3
        }
    },
    "data_loader": {
        "batch_size": 1,
        "shuffle": true
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 20,
        "cuda_device": 0
    }
}
