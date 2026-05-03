#!/usr/bin/env sh
set -eu

cargo fmt --all -- --check
CARGO_INCREMENTAL=0 cargo check --workspace
CARGO_INCREMENTAL=0 cargo test --workspace
