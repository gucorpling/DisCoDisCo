{
    "type": "character_encoding",
    "embedding": {
        "embedding_dim": 32,
        "vocab_namespace": "token_characters"
    },
    "encoder": {
        "type": "lstm",
        "input_size": 32,
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.2,
        "bidirectional": true
    }
}