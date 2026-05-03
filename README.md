# AionFM

Rust workspace skeleton for the AionFM temporal foundation model described in `../aionfm-spec`.

The workspace follows the layer boundaries from the implementation blueprint:

- `aionfm-data`: ingestion, normalization, patching, metadata, and streaming traits.
- `aionfm-tokenization`: residual descriptors, regime vocabularies, and quantizers.
- `aionfm-model`: dual-stream model contracts, config, embeddings, memory, heads, and a deterministic baseline model.
- `aionfm-training`: loss aggregation, trainer orchestration, checkpoints, and synthetic trajectories.
- `aionfm-adapter`: domain adapters, calibration, registry, and adaptation workflows.
- `aionfm-serving`: inference engine, scenario sampling, and monitoring hooks.
- `aionfm-utils`: shared contracts, error handling, schema types, and validation.

This is a compileable skeleton. It defines stable APIs and placeholder implementations for downstream repositories while leaving neural kernels, storage backends, and production serving internals to future implementation passes.

## Commands

```sh
cargo fmt --all
cargo check --workspace
cargo test --workspace
```

## Spec References

- Doc41: Implementation Blueprint
- Doc42: Example Forecast Output Schema
- Doc52: Corpus Storage Formats and Rust Integration
- Doc53: Directory Layout and Build System
- Doc54: Module Guidelines and API Design
- Doc55: Testing and CI/CD Guidelines
