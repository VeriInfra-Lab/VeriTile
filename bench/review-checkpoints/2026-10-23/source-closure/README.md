# Source closure repair

The selected `kv_cache_filling` delivery cannot yet be rebuilt from Git.
Its recorded consumer target list names 16 absent proof modules. The selected
public entries also refer to absent source snapshots under `.veritile-cache`.
See `missing-sources.json` for exact paths and the source commit under review.
This checkpoint does **not** renew the historical task acceptance or authorize
merging the full checkpoint.

## Repairs

- The two `chunked_cumsum_fwd` cached source references now point to tracked
  files with exactly the same pinned SHA-256 values. The ledger and its three
  current organization manifests select the same paths. Historical commands
  and receipts retain their original provenance.
- `TritonBenchGContracts` owns the `bench.repairs` module prefix, allowing Lake
  to discover transitive proof helpers without adding every historical repair
  entry to the default target list.
- Floating minimum reduction has a source parser, AST node, Real interpretation,
  opaque FP callback, precision propagation, safety, typing, flattening and
  locality cases. Unsupported callbacks and empty axes fail. The frontend
  explicitly rejects unsupported typed/discrete channels instead of silently
  casting them. `FPMinimumReduction.lean` checks source precision, callback
  dispatch and these failure cases.
- The six existing copy/INT8/INT4 programs are directly readable in
  `kv_cache_filling/Kernels.lean`. The previous scalar/model definitions are
  byte-for-byte preserved in `Model.lean`; compatibility source imports remain.
  `Proofs/SourcePreservation.lean` proves equality with both original macro
  specializations for every policy. It preserves the complete programs,
  including precision, casts, masks and stores.
- The copy execution proof still establishes successful execution, cache
  readback, exterior frame and equal final states for the two versions.
- Fresh compilation exposed kernel recursion failures in existing BF16
  cast and scalar-probe proofs. Their statements are unchanged; direct arithmetic bounds and
  a generic finite-decoding lemma avoid expanding large concrete bit patterns
  during proof checking.

These are repairs of available sources. They are not reconstructions of the
missing quantization execution, Real, native or backing-memory proofs. In
particular, the new readable source file is not substituted for a differently
hashed historical delivery.

## Validation and remaining work

`validation.json` records the commands, outcomes and log hashes. The clean
checkout starts with no `.lake/build` or `.veritile-cache`; only dependencies
from the pinned Lake manifest reuse their installed sources/builds. It does
not reuse another checkout's VeriTile artifacts.

The four original Python acceptance guards still fail on missing selected
sources. The public kernel build fails on missing `SharedTarget.lean` and
`SharedReadback.lean`. Those failures are retained in the logs, rather than
replaced with old receipts. Independent comparator replay is unavailable
because the official comparator is not installed in this environment.

To finish, recover the source snapshots listed in `missing-sources.json` from
the delivery producer, verify their pinned hashes, put authoritative sources
under the tracked task directory, and repair imports/selected paths. Then
rebuild the complete public dependency closure and produce new validation
receipts for the actual tracked source. Missing historical receipts must be
restored with their original hashes or explicitly superseded by newly run
checks; renaming a different source or fabricating a prior log is insufficient.

The deferred `matmul_leakyrelu` numerical work and the separately documented
cross-task import / `llama_ff_triton` limitations remain outside these focused
source repairs. No GPU experiment was rerun.
