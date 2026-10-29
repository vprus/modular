# MAX model architectures

This directory contains the model architectures supported by MAX pipelines. Each
architecture defines how MAX reconstructs models from Hugging Face as
performance-optimized MAX graphs.

If you'd like to contribute a new model, read [Contributing custom model
architectures](/max/docs/contributing-models.md).

The
[experimental Qwen3 embedding prefix cache](qwen3_embedding_modulev3/README.md)
is an explicit opt-in variant with 16-token matching and partial-page copying.
