from fast_transformers.builders import TransformerEncoderBuilder

fast_transformer = TransformerEncoderBuilder.from_kwargs(
    n_layers=8,
    n_heads=8,
    query_dimensions=96,
    value_dimensions=96,
    feed_forward_dimensions=3072,
    attention_type="full"
).get()

