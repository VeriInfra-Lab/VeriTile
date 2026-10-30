# Softmax FlagGems: contiguous FP32 repair

The original backward kernels leave masked loads undefined. Those arbitrary
padding values participate in the reduction and can change valid gradients.
The original wrapper also divides by zero for empty prefix/reduction dimensions;
an empty suffix can select the inner kernel and access an empty allocation.
The original Python remains unchanged.

`softmax_flaggems_repaired.py` adds explicit zero defaults to every backward
masked load, returns empty allocations before dimension division or launch, and
widens dimensions, program IDs, aranges, and pointer-offset arithmetic to int64.
It retains all four kernels, both branches in each kernel, all six backward
autotune configurations, the forward tile heuristic, and the original floating
arithmetic. Fusion remains enabled, including backward subtraction contraction.
The SMS query now occurs on the selected device and is passed explicitly to the
non-inner forward heuristic.

## Checked scope

Inputs and upstream gradients are strided contiguous FP32 CUDA Tensors with no
unresolved negative view bit, fewer than `2**31` elements, and a 4-byte-aligned
nonempty data pointer. Dim is a strict Python integer selecting an existing
axis, including negative axes. Dtype is omitted or explicitly FP32. Upstream
gradient shape/device must equal the saved output's shape/device. No finiteness
or zero-storage-offset restriction is imposed. Empty inputs bypass pointer
inspection and all launches. Fresh `empty_like` output and gradient allocations,
saved output context, and the `(gradient, None, None)` return are retained.

Contiguity is checked instead of materializing inputs. This scope rejects a
broadcast or otherwise noncontiguous upstream gradient, including gradients
that some ordinary autograd consumers generate. General-stride autograd support
is a later contract-review choice; this candidate states the narrower scope
explicitly. The contract does not certify the unchanged original wrapper.

## Proof entry points

- `Kernels.lean` holds the exact typed source. Forward arithmetic is reused from
  the original; repaired backward programs change only masked defaults.
- `Correct.softmax_autograd_real` constructs complete forward/backward source
  executions with independent unshifted softmax and conventional VJP results,
  actual saved-output dependence, returned metadata, and the whole memory frame.
- `FPEquiv.softmax_autograd_words` derives complete source execution and the
  selected native instruction graph for every output and gradient word.
- `FPEquiv.softmax_autograd_bytes` transports both returned regions to their
  allocated data pointers, gives all four bytes per element, preserves bytes
  outside both outputs, and preserves logical input cells. `Allocated` and
  `Binding.Separate` state ordinary allocator metadata/span/freshness facts.
- `Proofs/Contract.lean` proves exact guard/domain equivalence, including dynamic
  Tensor and strict integer classification. `Proofs/Pointers.lean` derives
  selected-configuration bounds, i32 primitive widening, full i64 address
  arithmetic, and active aligned four-byte spans. Padded lanes and post-loop
  carried offsets are included. Reverse writeback subtraction is exact.
- `Native.host_native_coverage` derives every actual forward/backward store
  event, including repeated tuning trials, and excludes exterior stores.
  `Binding.store_bytes` proves the four-byte effect of an individual source
  store. `Inputs.input_binding_identity` proves direct input leaf binding makes
  no copy; the stronger frame preserves all cells outside the two outputs.

The mathematical and FP roots are independent. The complete source closure
retains the original real proofs and the historical structural FP proofs under
`Proofs/`; neither root assumes a whole output, success, reduction identity,
addressing result, or store coverage. Native reduction trees and local fusion
choices are shape-selected compiler parameters. FP32 exp is explicitly lowered
to multiplication by encoded `0x3fb8aa3b` followed by `ex2.approx`; FP32 casts,
load barriers, product operand swaps, FMA contractions, and negated-factor FMA
for subtraction remain visible.

The Tensor boundary binds the logical forward input and upstream-gradient entry
snapshots to their actual contiguous pointer reads, and binds the two fresh
allocator results to output descriptors. Forward and backward occur at distinct
API entry times; an upstream gradient aliasing saved output must satisfy the
ordinary read binding at backward entry. Abstract region names do not assert
that read tensors occupy disjoint physical allocations. Tensor metadata,
contiguity, allocation, byte packing/load-store semantics, selected compiler
lowering and successful race-free runtime execution remain ordinary external
contracts; this is not a verified Python, Torch, Triton, or GPU implementation.

## Numerical and runtime boundary

`output_exact_if_local` and `gradient_exact_if_local` condition only on local
FP32 add/sub/mul/FMA, full division, maximum, and exp2 comparisons. Their recursive
domain records actual intermediate operands, including reduction nodes, FMA
operands and exponential scaling. Finite input leaves alone are insufficient.
The reused numerical-port bundle also contains a logarithm port, unused by this
source. Structural word/byte results allow every FP32 word and impose none of
these numerical conditions. GPU numerical validation is deferred; no native
conformance or FP-to-real equality is claimed.

The shared Triton launcher large-grid product compatibility issue remains
separately deferred in `bench/TODO.md`. Source indexing, active pointers,
padding, reverse coverage and autotune stores are proved here rather than
classified as numerical obligations.

## Validation

Targeted Lean 4.29.0 builds pass with standard-base axiom audits and noncircular
public specifications; there is no `sorry` or new axiom. Run:

```sh
python bench/repairs/tritonbench_g/softmax_flaggems/check_contract.py
```

The checker covers 22 forward/backward cases, 21 rejections and five original
counterexamples, with 51 source launches. It executes all four kernel families,
both branches, all six backward configurations, large forward loops, negative
and arbitrary axes, empty dimensions, explicit dtype and saved-output behavior.
Its AST comparison removes only repaired integer widening, explicit zero
padding and the extra SMS argument. CUDA/Triton launches are mocked; numerical
comparison is CPU evidence only.

Offline Triton 3.7.1 compilation passed 41 SM80 profiles, including forward
one/multiple tiles and all six backward configurations at N=1, 3 and 1025.
It confirmed int64 address operations, exp2 lowering, reduction/product FMA,
subtraction FMA in the single-row non-inner case, and selected multiplication
operand reversal. Artifacts are `/tmp/veritile-flaggems-lowering/`; reproducer is
`/tmp/veritile-flaggems-lowering.py`. CUDA execution was unavailable.

Original Python SHA256:
`11f0c747af151465286ea9e8821932f80b71513fbb2d343a304ba54f4e8013f4`.
