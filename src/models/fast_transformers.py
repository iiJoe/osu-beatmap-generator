from fast_transformers.builders import TransformerEncoderBuilder

fast_transformer = TransformerEncoderBuilder.from_kwargs(
    n_layers=6,
    n_heads=8,
    query_dimensions=16,
    value_dimensions=16,
    feed_forward_dimensions=512,
    attention_type="linear"
).get()
