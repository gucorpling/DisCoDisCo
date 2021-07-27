local transformer_model_name = std.extVar("EMBEDDING_MODEL_NAME");

local embedding_dim = std.parseInt(std.extVar("EMBEDDING_DIMS"));  # uniquely determined by transformer_model

local features = {
  "nuc_children": {"source_key": "nuc_children"},
  "genre": {"source_key": "genre", "label_namespace": "genre"},
  "u1_discontinuous": {"source_key": "u1_discontinuous", "label_namespace": "discontinuous"},
  "u2_discontinuous": {"source_key": "u2_discontinuous", "label_namespace": "discontinuous"},
  "u1_issent": {"source_key": "u1_issent", "label_namespace": "issent"},
  "u2_issent": {"source_key": "u2_issent", "label_namespace": "issent"},
  "length_ratio": {"source_key": "length_ratio"},
  "same_speaker": {"source_key": "same_speaker", "label_namespace": "same_speaker"},
  "doclen": {"source_key": "doclen"},
  "u1_position": {"source_key": "u1_position"},
  "u2_position": {"source_key": "u2_position"},
  "distance": {"source_key": "distance"},
  "lex_overlap_length": {"source_key": "lex_overlap_length"}
};

{
  "dataset_reader" : {
    "type": "disrpt_2021_rel_flair_clone",
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model_name,
        "max_length": 511
      }
    },
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model_name
    },
    "features": features
  },
  "train_data_path": std.extVar("TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("VALIDATION_DATA_PATH"),
  "model": {
    "type": "disrpt_2021_flair_clone",
    "embedder": {
      "type": "featureful_bert",
      "model_name": transformer_model_name,
      "max_length": 511,
      "train_parameters": true,
      "last_layer_only": true
    },
    "seq2vec_encoder": {
        "type": "bert_pooler",
        "pretrained_model": transformer_model_name
    },
    "feature_dropout": 0.0,
    "features": features,
  },
  "data_loader": {
    "batch_size": 4,
    "shuffle": true
  },
  "trainer": {
    "num_epochs": 100,
    "patience": 12,
    "optimizer": {
      "type": "adamw",
      "lr": 2e-5,
      #"weight_decay": 0.05,
      #"betas": [0.9, 0.99],
      #"parameter_groups": [
      #  [[".*embedder.*transformer.*"], {"lr": 2e-5}]
      #],
    },
    #"learning_rate_scheduler": {
    #  "type": "slanted_triangular",
    #  "num_epochs": 50,
    #  "cut_frac": 0.1,
    #},
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.6,
      "mode": "max",
      "patience": 2,
      "verbose": true,
      "min_lr": 5e-7
    },
    //"learning_rate_scheduler": {
    //  "type": "cosine",
    //  "t_initial": 5,
    //},
    //"validation_metric": "+relation_accuracy"
  }
}
