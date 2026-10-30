# SGMV expand slice

The checked repair calls the retained kernel directly on the original CUDA
buffers. X and Y are FP32, W is FP16, and each metadata array independently
uses int32 or int64. Both add modes, K=0, K/M/N tails, singleton 4D weights,
aligned storage-offset views, read/read aliases and overlapping segments are
supported. The supplied Y object/backing is preserved and the function returns
`None`.

The wrapper validates contiguous, nonlazy layouts, ranks, dimensions, device,
scalar types, slice capacity, metadata contents, native argument/grid limits,
formed signed-64 byte offsets and pointer alignment. Y's backing must be
separate from every read operand. It uses the actual CUDA device context,
canonical strides and a singleton squeeze alias. It does not allocate tensor
snapshots, materialize operands or copy output back. Host metadata `tolist`
calls validate and schedule the original metadata buffers. Three metadata
arrays are checked before the first launch. Zero batches skip inspection;
zero columns still validate metadata. Zero/sentinel segments skip launches.

Actual segment lengths determine the grid. `max_seq_length` is a nonnegative
hint and cannot truncate coverage. Batches launch serially in original order:
overlapping replacement is last-wins, and overlapping addition preserves that
order. Read operands remain unchanged throughout.

The four public Lean entries are `Kernels`, `Contract`, `Correct`, and `FPEquiv`.
All implementation/proof helpers are under `Proofs`. `Correct.checked_real`
proves complete typed-source execution and the independent ordinary K-term
contractions, folded in segment order. `FPEquiv.checked_fp` and
`FPEquiv.checked_caller_bytes` prove complete native-source execution, exact
four-byte destinations and the complement frame from the same executable
checks and actual original-storage observations. The native implementation
bytes come from final source memory. The reference is independent.

`Proofs/Direct/Bounds` derives all clamped lane offsets, including masked lanes,
from checks. `Physical` derives native pointer addresses and metadata spans,
whole read-backing preservation, empty writes and arbitrary failed-write
subset/order frames. Failure may leave issued destinations changed. The Real
and FP entries are independent imports. Historical host/snapshot statements
remain under `Proofs/LegacyCorrect` and `Proofs/LegacyFPEquiv` for proof credit;
the new public results use no snapshot/prepared-host success premise.

The historical `check_contract.py` runs the actual production host/kernel ASTs
on CPU storage, checks historical source hashes and the unchanged kernel AST, and covers 33
complete calls, 57 rejections, serial overlaps, all metadata widths, direct
identities, byte frames, context restoration and partial launch failure.
This checks control/storage behavior; it does not establish GPU arithmetic
conformance. See `NUMERICS.md` for the local native boundary.

The 48 complete source definitions in `Kernels.lean` now use ordinary
`triton { ... }` blocks. Explicit terminal casts and stores in each addition
branch spell out the same statements formerly expanded by the task quotation
adapter. Every complete old and new function is equal by `rfl`; source names,
ABIs, precision and selection are preserved. Historical source pin files retain
the quotation's original hash, and the campaign preservation check records the
current source connection.

The selected `ByteSeparated` repair now has a complete actual implementation
pair. Its candidate reuses the existing `offset_m`/`offset_n` when forming output
coordinates, removing duplicated integer tile-vector construction. Its main
`FPEquiv.sgmv_grid_equiv` retains all eight metadata ABIs, full K loops,
ordered overlaps, both add modes, all original source-format words, both safety
and success results, all outputs and exterior frames, with no experimental
numerical rules. See `ByteSeparated/README.md` for the candidate and proof scope.
