# Triton source and proof checks

Run `bash scripts/check-core-examples.sh` from the VeriTile root to check the
three worked examples used in the library CI. For an individual example, use
`python3 -m scripts.bench.generate_jit_correspondence bench/examples/VectorAdd`.
The tool discovers and checks source pairs without writing a JSON manifest.

Corpus inventory, per-task validation, and the daily benchmark audit are
maintained in [VeriTileBench](https://github.com/VeriInfra-Lab/VeriTileBench).
