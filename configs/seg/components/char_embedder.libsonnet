{
    "type": "character_encoding",
    "embedding": {
        "embedding_dim": 40,
        "vocab_namespace": "token_characters"
    },
    "encoder": {
        //"type": "cnn",
        //"embedding_dim": 40,
        //"num_filters": 80,
        //"ngram_filter_sizes": [2,3,4,5],
        //"conv_layer_activation": "leaky_relu"
        "type": "lstm",
        "input_size": 40,
        "hidden_size": 80,
        "num_layers": 1,
        "dropout": 0.2,
        "bidirectional": true
    }
}