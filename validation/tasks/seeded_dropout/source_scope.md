# Source and theorem scope

## Pin and preserved source

- Upstream: TritonBench-G commit `603e28a`, `data/TritonBench_G_v1/seeded_dropout.py`.
- Exact Python SHA256: `2fd4f4e6aff35d5720156b9cf046f2d112173a431d886106a6441382f63ab570`.
- Kernel: `_seeded_dropout`; the wrapper allocates `torch.empty_like(x)`, checks
  contiguity, passes runtime N/p/seed, and launches a cdiv grid with B=1024.
- The source body is preserved, including `tl.load(..., mask=mask)` without
  `other`, `tl.rand(seed, offsets)`, strict greater-than, the eagerly evaluated
  divide in `tl.where`, and the final mask.
- The only candidate rewrite commutes the integer block-start product.

The explicit `seed_arg` binding materializes the same external seed, with
`tl.source_integer` recording its original ABI. `BLOCK_SIZE_arg` materializes
B through the rooted `Random.i32ArangeSize` identity and a proved arange-domain
witness. Neither adapter computes a different value. The explicit destination
cast represents the implicit conversion at the native store. These adapters
need exact source-profile mappings and full RHS validation in production gates.

## Formal parameters

| Parameter | Formal scope |
|---|---|
| Storage | fp32, fp16, bf16, fp64, f8e4, f8e5 |
| Seed | original runtime value in i32/u32/i64/u64 ABI |
| p | original runtime real parameter; no probability or nonzero-denominator restriction |
| B | symbolic `2^k`, with `k ≤ 20` |
| N | natural size, with `N < 2^31` for complete real execution |
| Program index | `pid < (N+B-1)/B`; N=0 has no launched programs |
| RNG | fixed Philox, i32 offsets, 10 rounds, canonical fp32 conversion and scaling |

The i32-domain proofs cover all launched lanes, including masked-off tails.
They do not impose `N+B-1 < 2^31`: the host grid expression uses Python integer
arithmetic. They prove the actual signed 32-bit product/addition/mask agree with
the mathematical indices in the admitted domain.

## Execution and memory

`Algorithm.source_eq` connects the complete erased source to its checked AST
normal form. `Execution.exec_storeBlock` proves the complete final state equals
the canonical writeback over the initial input snapshot after the exact source
register assignments. No preinitialized-register or assumed-execution premise
stands in for the source run.

`Safety.trace_safe` proves the addresses actually accessed are within the
provided allocation bounds. Inactive lanes perform no memory accesses, while
the source still generates their random values and evaluates eager operands.
`RealIO.Implements` composes actual execution and safety through the generic
flat-memory bridge. Active outputs are exact `MemCell.real` cells. Every other
cell is preserved with its tag and pointer relocation, including unallocated
flat addresses. All non-memory state is retained by the exact execution model.

`RealSurface.realizes_frame` derives the standard
`ComputeCorrect.RealizesFrame_without_Rounding (α := MemCell)` interface from
that successful complete-source execution. Its exact masked write map and
input-only expected formula cover every active lane, while its frame preserves
every other tagged cell. The public original/candidate specifications conjoin
this standard region interface with the unchanged successful finite-flat
contract. No existing KernelIO skin packages the runtime seed/storage profile.

The current generic whole-state flatten theorem requires zero `undef`; that
premise is explicit in the public real contract. It does not rewrite the
source load padding. Per-program input/output aliases are admitted; concurrent
program race freedom for aliased buffers is not claimed. Lane/program coverage
and output-index disjointness are separate arithmetic theorems.

## Floating-point claim

`FPProof.whole_state` proves equality of complete typed-source execution for
every structural backend algebra and initial state, including failure. The
public numerical assumption list and operand-domain list are empty. Original
and candidate share the whole floating suffix, including fixed canonical
random bits, original comparison, eager division, and storage conversion.
This is implementation equivalence, not equality of physical floating results
to the real formula. It does not delete p=1 division or rely on total-real
arithmetic as a floating rewrite law.

The separate `TypedExecution` consistency witness starts with empty registers
and N=B=1, PID=0, so its sole lane is active. It proves complete original and
candidate runs succeed for all six storage × four seed profiles, arbitrary
runtime p and every original seed in the selected ABI. The abstract helper
requires supported total comparisons. `nonempty_pair` discharges that condition
using an explicit `Structural.Algebra ℝ` and the public relation's standard
default-fp32 engine. Its real arithmetic, identity cast/load and finite bit
decoder are consistency interpretations; they add no numerical law or premise
to the public equivalence and provide no IEEE acceptance claim. The source RNG
remains canonical, with its actual fp32 bit payload. This witness is scoped to
one active lane; the public real and equivalence specifications remain symbolic
in the legal block size.

## Native and integration limits

Separate offline Triton 3.7.1/sm90 compilation at the exact wrapper B=1024
accepted all six storage × four seed ABI profiles with dynamic N:i32/p:fp32/seed.
The original JIT AST is unchanged; only that function was extracted, avoiding
the Python file's CUDA test invocation. Compiler artifacts and hashes are in
`/tmp/veritile-seeded-dropout-native-b1024/compile_receipt.json`.

`b32_comparison_receipt.json` records exact fp32 TTIR equality for the four seed
ABIs after removing debug locations and normalizing block-size extents/constants.
All 24 B1024 artifacts retain i32 offsets and signed masks, omitted load padding,
division before the output select, and fp32 Philox conversion/scaling and
comparison. fp64 output division uses f64 after extending the fp32 denominator;
the other profiles divide in f32 and apply the native destination conversion.
The wrapper's tested Python seeds 123/512 infer i32 and p=0.5/0.3 infer fp32.

These are compiler observations, not theorem assumptions or hardware validation.
No GPU was run. Six floating profiles do not characterize all accepted torch
dtypes. Host runtime behavior, end-to-end corpus acceptance, and independent
source-audit replay remain integration tasks.
