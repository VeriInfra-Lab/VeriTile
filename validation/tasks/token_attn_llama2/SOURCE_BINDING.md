# Source binding

Original source SHA256: `3dccffd5a31077851127ee70f8ec64136fdc0872ef29399b3bf74a8223023d5f`.
The original file is unchanged. Its `_fwd_kernel_token_att1` AST equals the
candidate kernel AST exactly. The wrapper adds only checked admission, an
empty return and the owning CUDA device context.

The six buffers, packed suffix gather, grouped head division, zero-or-one loop,
masked K/Loc loads, Q load, FP32 products, dimension reduction, scale multiply,
masked output stores, grid and warp/stage selection are represented in the
full structural source proof. Host metadata reads validate the same arrays
passed to the single kernel launch, under the ordinary no-concurrent-mutation
contract. Original Tensor identity and caller storage are retained.

The Real theorem erases floating precision and uses ideal sqrt. The FP theorem
uses actual host scale words 0x3e800000, 0x3e3504f3, 0x3e000000 and 0x3db504f3.
Local instruction conformance is to rounded FP32 operations, including IEEE
exceptional values; no FP-to-Real accuracy statement is implied. Complete tree
plans are structural parameters. Pinned compiler evidence is conditional on its
explicit Binder metadata, and is not a proof of an entire compiler.

Validation artifacts for this run: `/tmp/veritile-llama2-final-build.log`,
`/tmp/veritile-token-llama2-check.py`, and
`/tmp/veritile-token-attention-lowering/report.json` (eight entries for this task).

The checked public `checked_call` / `checked_fp_call` endpoints compose empty and launch validation. Empty dispatch preserves the supplied descriptor and current memory without canonical inactive strides. Live calls reject mixed FP32/int32 input-byte overlap while retaining same-dtype input aliases. The regression checker uses a fixed random seed; its comparison against a double-precision reference is an approximation check, separate from the word-level FP theorem.
