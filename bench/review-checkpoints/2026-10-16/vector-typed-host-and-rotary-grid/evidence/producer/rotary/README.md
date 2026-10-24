# RotaryEmb host/grid producer candidate

This candidate extends the retained original 34-statement exact-real-erased
source proof to a complete finite host grid. It is partial progress pending
independent review and repository/catalog integration, not full acceptance or
rounded/native execution certification.

`receipt.json` records the eight actual successful mapped-module compiler runs,
the public statement, exact source/import transformations, dependency manifest,
63 original-wrapper AST probes, and remaining scope. `dependencies.json` records
the actual imported module inventory with SHA-256 hashes of sources, oleans and
their private/server companions. The output overlay uses support symlinks into
the existing build tree; it is not a standalone copy of all dependencies.

The public entry is
`VeriTile.Bench.TritonBenchG.RotaryEmb.HostGrid.rotary_emb_host_grid_correctness`
in `overlay/bench/tritonbench_g/rotary_emb/HostCorrect.lean`. Its independent
`Plan.Output` refers to logical tensor cells, initial memory, and write frames.
`Plan.run` executes each complete original kernel on preceding memory with fresh
registers. `run_correct` proves composition for any duplicate-free PID order;
`grid_correct` instantiates an explicit complete grid schedule. Source access
safety is checked at the actual intermediate states. Extra K heads beyond
`4 * ceil(HEAD_Q/4)` and odd/partial channel tails are preserved as raw cells.

The exact-real model uses four symbolic storage selections only to choose the
retained source before dtype erasure. It is not a rounded typed theorem. No
numerical assumptions or deferred floating-point equality laws are added.
The domain explicitly requires matching sequence shapes, nonnegative converted
dimension, natural strides, sufficient logical widths, bounds on active source
addresses, full tensor-layout injectivity, and separation between output and
other regions. Native dtype/arange/resource admission, finite machine index
overflow, arbitrary aliasing, byte-level execution and rounded arithmetic remain
separate obligations. Cos and Sin may alias; both use Cos strides as in source.

The concrete witness has 17 sequences, five Q heads, ten K heads, full width
seven, converted rotary width five, nonunit strides and four programs. Its
domain and successful execution are proved without launch-success premises.
The 63 AST probes execute the unchanged Python host FunctionDef with shape,
stride, cdiv and launch instrumentation; Python float multiplication/int
conversion are the actual interpreter primitives. They cover all launch
arguments, both warp branches, empty grids, negative/odd/partial dimensions,
assertion failures and NaN/Inf conversion. They do not execute the GPU kernel.

To integrate, copy only the eight files listed under `transformations` in
`receipt.json`, retaining their mapped relative paths. `SupportInventory.lean`
is an audit helper and can remain review-only. Existing `Correct.lean` and the
kernel/source files are unchanged. Public specification/catalog registration
has intentionally not been performed by the producer. All exact local import
rewrites are listed in `import_mapping`; the mapped files have compiled.

The source-parameterized compile commands and LEAN_PATH are in the receipt.
`producer-evidence/attempts` retains later failed attempts. The initial broken
HostExecution source was recovered byte-for-byte and replayed in
`failed-replay`; the original raw failure log was overwritten before the
preservation instruction arrived, disclosed explicitly in
`producer-evidence/attempts/preexisting-recovery.json`. The first mapped-overlay
namespace resolution failure is retained in the sibling `../frozen` directory.
