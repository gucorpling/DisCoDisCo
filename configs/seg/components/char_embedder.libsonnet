{
    "type": "character_encoding",
    "embedding": {
        "embedding_dim": 25,
        "vocab_namespace": "token_characters"
    },
    "encoder": {
        "type": "lstm",
        "input_size": 25,
        "hidden_size": 80,
        "num_layers": 1,
        "dropout": 0.2,
        "bidirectional": true
    }
}