# Architecture

This repository is organized around AionFM's layered implementation blueprint.

Data loaders produce normalized patches and metadata. Tokenization converts patches into residual descriptors and hierarchical regime tokens. The model layer fuses continuous value patches and discrete regimes through dual-stream contracts. Training, adaptation, and serving consume those shared contracts without depending on concrete neural backends.
