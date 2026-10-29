These text fixtures preserve the actual seeded-dropout source checkpoint
`/tmp/veritile-seeded-dropout-port/builds/20261013T095124881199Z/src/bench/tritonbench_g/seeded_dropout`.
The source Python is pinned to upstream revision `603e28a`.

- `seeded_dropout.py.txt`: `2fd4f4e6aff35d5720156b9cf046f2d112173a431d886106a6441382f63ab570`.
- `Kernels.lean.txt`: `d22dd1a19598ac8576e225a32dec2c70aea618bcffd374e84ed29103bf60dd30`.
- `RandomProfile.lean.txt`: `9001fbfeecd8bd2a68428ca9f49a72879b7bd1e8ebd8b14da8000590e829e140`,
  copied from the frozen random frontend's `VeriTile/Triton/DSL/RandomProfile.lean`.
- `RandomProfile-20261013.lean.txt`: `f7704622063dbaed4d3de60e7a9a49d3aaecffcb1949b70c5fbbc3d8f0c02f37`,
  the reviewed frontend after modular unsigned casts and augmented-assignment
  provenance preservation. The `i32ArangeSize` identity definition and its
  recognizer are unchanged. Tests use this current adapter checkpoint; the
  earlier snapshot above remains available as historical evidence. Lean and
  axiom-check logs are in
  the current signed/unsigned cast regression fixtures.
- `RandomProfile-20261013-direct.lean.txt`: `37e7fadbcdfe41dc76c68638ab3ed4215c187752b097f38958d03f1255054467`,
  the reviewed direct-Philox, explicit helper-binding and unsigned-arithmetic
  frontend. Its identity adapter definitions and recognizers are unchanged.
  Current tests use this version; both earlier adapter files remain historical
  snapshots. This source pin does not assert native or independent proof replay.

The three proof text files are the transitive local imports needed for production
source discovery; they are not new compilation receipts. The original compilation
receipt SHA-256 is `2e6504fc9407afcd740987ee70702caddb3bcd4ba480114324eb18d72d122e3a`.

Tests copy the unmodified source into a temporary single-entry legacy layout,
add explicit comparison metadata, and use the production source-view gates.
They do not create `Correct.lean` or `FPEquiv.lean` placeholders, publish a port,
compile native code, or assert whole-corpus correctness. The fixture adapter is
used only as read-only provenance evidence; production profiles check the actual
repository library path and exact reviewed hash.

- `RandomProfile-20261014-signed.lean.txt`: `61f31027a53ea7305b18341f0c7dced51c9407589bf4fe16b1c7bfded7859d04`. Adds i64-to-i32 source provenance only after actual signed low-word cast lowering. Identity adapters/recognizers are unchanged; older fixtures are retained.
