# Checked stream-K repair

Reference responsibilities (2026-10-23): `Whole001/Reference.lean` retains the independent exact K-term Real matrix product. Chunk witnesses, the ordered atomic `streamValue` graph and its complete source connection are in `Whole001/Proofs/Reference.lean`; the selected `streamk_matmul_tile_checked` FP comparison still requires those proofs.

The candidate is [`Whole001/streamk_matmul_repaired.py`](Whole001/streamk_matmul_repaired.py).
It retains the previously accepted Whole001 three-launch repair and its proofs.
All six kernel ASTs match that accepted baseline. The new changes are explicit
host admission checks, canonical contiguous strides, source-only extraction,
and direct checked input, allocation, execution, and returned-memory bridges.
The original benchmark Python remains unchanged (SHA-256
`794c7aa68254a0cc1caf974b5ed806ce8c884e9f7c9c2c36d68c67d5b654a4f3`).

## Original defect and repair behavior

The original source uses an inter-CTA lock/publication protocol. Its old proof
package records a blocked poll when the required publisher does not run,
unmasked tail reads/writes, observable full-tile FP16 rounding, and order-sensitive
FP32 atomic additions. See the original task README for the concrete witnesses.

Whole001 replaces that publication protocol with ordered launches on one stream:

1. `first_wave` stores each terminal chunk to C and each nonterminal chunk to
   the owning program's FP32 scratch tile.
2. `reduce_streamk` reads only scratch tiles whose producer is established by
   the integer partition, scanning programs in ascending order for each tile.
3. `full_tiles` writes remaining ordinary tiles, preserving the original
   explicit FP16 narrowing followed by FP32 storage.

All input/output tails are masked. Pointer and iteration arithmetic is widened
where required by this accepted repair. No spin lock, CTA residency, or fairness
premise is needed. Scratch exists only when stream tiles exist. Empty output
allocates C and returns without launching; K=0 uses the ordinary zero-accumulator
full-tile path. Nonpositive program counts select ordinary blocking.

The repair keeps FP32 input/scratch/output storage, the original fixed-MMA TF32
lowering and FP32 accumulator, caller block sizes, `two_tiles`, stages and warps.
It keeps the full-tile FP16 round trip and the ordered atomic recurrence. The
native theorem does not equate stream and full arithmetic through associativity
or an assumed split/fused-dot equation.

## Checked domain

The wrapper accepts the original tested FP32 storage profile: rank-two strided,
contiguous, resolved tensors on the same CUDA device with matching K dimensions.
Input views may share storage. It does not copy, flatten, normalize, or reshape
inputs. Canonical strides `(K,1,N,1,max(N,1),1)` preserve contiguous singleton
semantics without depending on irrelevant singleton stride metadata.

Python checks integer launch parameters, excluding Boolean integers; a Boolean
`two_tiles`; positive power-of-two blocks at most `2^20`; BK at least 8 and a
multiple of 8; positive stages; and supported warp-count syntax. For nonempty
output it also checks:

- All source i32 ceiling numerators, grouped width, tile count and nonnegative
  program count are below `2^31`.
- Stream iteration work plus one tile's iterations is below `2^63`.
- Four times each allocation size and conservative pointer envelope is below
  `2^63`, including inactive tails and terminal pointer advances.
- Every nonempty input pointer is aligned to four bytes and its complete byte
  window fits the 64-bit address space.

`Contract.checkInputs_iff` proves checker acceptance is exactly the declared
contract. `Envelope.valid` derives the inherited source bounds from those
checks. These bounds are not embedded in ordinary Tensor or allocation laws.
Successful compilation/device dispatch is explicit in `Host.Ready`, for exactly
`Plan.launches` with the actual tensor objects and unchanged configuration.
Compiler/device rejection and allocation failure remain ordinary API failures;
positive power-of-two blocks alone are not claimed to guarantee compilation on
every GPU.

## Public results

- `Correct.matmul_checked_real` takes an unrestricted `Nat → ℝ` physical
  element-start reader. It follows the checked host allocation/dispatch path,
  returns the actual C object, and proves every output equals the independent
  full-K sum over the original logical input views. It includes output metadata,
  physical input-storage/exterior frames, derived source bounds and physical operand conformance.
- `FPEquiv.matmul_checked_if_local` follows the actual native store/atomic byte
  executor. It proves every returned FP32 word and each of its four little-endian
  bytes, address representability, original input-storage preservation, exterior
  pre-call byte frame, output metadata, and returned object identity. Its target
  is the independent terminal-chunk/ascending-program atomic recurrence for
  stream tiles and the original half-rounded recurrence for full tiles.

`ValueMemory.call_render` proves the arbitrary-Real physical store/atomic executor
against the complete source schedule, without restricting input Reals to decoded
finite payloads. `Index.nativeBlock_eq` composes signed masks and finite i64 input addresses through
every fixed MMA8 block. `ByteExecution.step_render` proves stores and atomic
read-modify-write against current C/scratch bytes. `ByteExecution.call_render`
composes them through arbitrary first/full store permutations and all reducer
CTA interleavings preserving each CTA's instruction order. Optional scratch has
no allocation, binding, or frame footprint when inactive.

`Kernels.lean` imports source definitions only. The new original
`HostDefinitions.lean` contains source host definitions extracted without changing
their names. `Whole001/Foundation.lean` and `SourceDefinitions.lean` contain the
source finite execution definitions; historical theorem modules retain the
partition, publication, geometry, reduction, reassembly and scheduling proofs.
`FPEquiv.lean` does not import `Correct.lean`.

The remaining numerical boundary is `Native.Numerical`: FP32 zero, one fixed
K8 TF32 MMA, scalar FP32-to-FP16 conversion, scalar FP16-to-FP32 conversion, and
scalar FP32 atomic add. It contains no variable-shape dot, whole-output equation,
layout/schedule premise, or split/fused equivalence. Ordinary Torch allocation,
compiler lowering and CUDA same-stream dispatch remain explicit execution
boundaries. No GPU numerical validation is claimed.

## Validation

- 48 distinct original/candidate Lean modules passed: the complete 46-module
  build plus the changed seven-module closure after adding the two ideal-memory
  modules. All 161 distinct axiom checks use only the standard base; zero errors.
- `python bench/repairs/tritonbench_g/streamk_matmul/check_contract.py`: 36
  rejection controls, 50 actual interpreted wrapper calls, 55 kernel launches.
  Checks cover the four original block profiles, both `two_tiles` values,
  split chunks, tail tiles, K=0, empty M/N, nonpositive P, P beyond work, aliased
  inputs, irrelevant singleton strides, output identity, input values and
  exterior allocation sentinels. Kernel AST hashes pin the accepted repair.
- `python bench/repairs/tritonbench_g/streamk_matmul/Proofs/compile_candidate.py
  --output /tmp/veritile-streamk-lowering`: 15 offline SM80 compilations across
  all original profiles and an additional large pointer/iteration profile.
  Audits i64 tensor pointer offsets, accumulator-fed K8 TF32 MMA, absence of a
  separate floating add, half conversion only in full tiles, and native FP32
  atomic addition in the reducer. Artifact hashes are recorded in `result.json`.

The installed environment has Torch 2.13.0+cu130 and Triton 3.7.1. Interpreter
checks run on CPU; offline compilation emits SM80 code. CUDA execution was not
available. Interpreter values were chosen to isolate indexing and memory behavior
from the unproved native numerical correspondences.
