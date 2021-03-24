{
    "type": "character_encoding",
    "embedding": {
        "embedding_dim": 40,
        "vocab_namespace": "token_characters"
    },
    "encoder": {
        "type": "lstm",
        "input_size": 40,
        "hidden_size": 40,
        "num_layers": 1,
        "dropout": 0.2,
        "bidirectional": true
    }
}