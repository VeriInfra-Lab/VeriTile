# Adam-named Lion: kernel correctness and native graph relation

The selected source implements **Lion**, with one momentum buffer and a sign
step. On entry-state logical elements, the independent Real formulas are:

```text
p' = (1 - lr*wd)*p - lr*sign(beta1*m + (1-beta1)*g)
m' = beta2*m + (1-beta2)*g
```

Parameter and momentum are updated in place. Gradient is read-only. Real is a
mathematical interpretation, not an assertion of exact FP32 arithmetic. The
source does not constrain tensor dtype; FP32 is the selected default, and the
checked repair requires all three tensors to be FP32.

## Selected source and input contract

The original benchmark is unchanged, SHA-256
`27d4f15342638339b2daea53f43752e30a28114a39d1b8dcdaf5e29b89072f45`.
Its wrapper admits arbitrary strided views to a flat-address kernel. For
example, a transposed parameter tensor and a gradient with a different logical
flattening order update different logical coordinates. `check_contract.py`
reproduces this defect; original correctness is not claimed on that domain.

The separately selected `adam_update_triton_repaired.py` retains the existing
kernel arithmetic, both autotune configurations and scalar normalization.
The Python implementation is unchanged. The compared candidate reuses the
benchmark's existing `update_commuted_kernel` transformation at the repaired
runtime scalar ABI: only `diff * beta2 + grad` becomes `grad + diff * beta2`.

`Contract.lean` matches the Python tensor guards:

- P, G and M are strided, contiguous FP32 CUDA Tensors on one device, with no
  unresolved negative bits. Shape/rank equality is not required.
- G and M each cover P's element count.
- Empty P returns before pointer or scalar checks.
- For nonempty P, all three actual pointers are divisible by 16 and the three
  full logical byte spans are pairwise disjoint. Disjoint offset views within
  shared storage remain valid.

The existing 16-byte guard selects the reviewed aligned compiler profile. It
is stronger than the 4-byte requirement of scalar `.b32` accesses; `.v4.b32`
requires 16-byte alignment. This is an explicit selected-domain restriction,
not a claim that all less-aligned FP32 tensors are invalid. The shared
[load/store audit](../add_example/LOAD_STORE_AUDIT.md) records the modeling
basis. This revision adds no guard, input copy, dtype variant, finite-value
condition or global size cap.

The four scalar arguments remain runtime FP32 ABI values. The retained Python
helper accepts bool/int/float payloads and their numeric subclasses, or one
exact `tl.constexpr` numeric wrapper. Integer conversion to Python binary64
and binary64 conversion to FP32 remain distinct API boundaries. Kernel proofs
start from the resulting scalar words; they do not encode arbitrary FP32
words as Real literals or assert a direct integer-to-FP32 rounding identity.

## Public kernel proofs

`Kernels.lean` contains both complete, ordinary DSL quotes. Four nonexecuting
`tl.param ... : tl.float32` declarations describe the actual runtime scalar
registers; no dummy assignment or body dropping is used. Both sources execute
the complete 19-statement body. `engine` fixes default precision to FP32.
The source's implicit bool-to-FP32 multiplication is written explicitly as
`where(can_update, 1, 0)`; it is not replaced with a Real-only mask identity.
The focused AST check compares every body statement with Python. The historical
`mathematicalSource` adapter is used only by preserved grid helpers.

- `Correct.lion_kernel_correctness` states `lionIO ⊨` the two independent
  parameter and momentum formulas. `IO.lean` declares the P/G/M input windows,
  the in-place P/M output windows, the active mask and the four runtime scalar
  register bindings. The shared `ProgramIO` judgment proves actual-source
  access safety, successful execution, typed outputs and the complete exterior
  cell frame. It includes zero and either sign branch, empty masks and arbitrary
  tails.
- `FPEquiv.lion_kernel_equiv` states the actual two-program
  `original ≡[[]] candidate`. Each program contains the complete quoted source
  in a `ProgramIO.Implementation`. Every permutation of the complete grid is
  supported, for every positive block size (including both retained autotune
  configurations), arbitrary total/chunk sizes, empty grids and tails. Both
  actual schedules are access-safe and terminate; every typed P/M output cell
  agrees after native observation, and each entire exterior cell frame is exact.
  The four scalar words and all input words are arbitrary FP32 bit patterns.
- `TensorBridge.lion_checked_views_correctness` composes this result with the shared
  Tensor storage reader. Nonzero storage offsets and direct logical reads are
  derived from checked metadata and ordinary Tensor API laws.
- The ordinary `FPEquiv.lion_kernel_native_equiv` theorem, preserved in
  `Misc/NativeCorrect.lean`, uses the same IO layout and actual source
  to state the independent ordered native recurrence below. Both output word
  cells, actual-source access safety and the complete exterior regional frame
  are included. Inputs are source-word leaves, preventing contraction across
  a load.
- `TensorBridge.lion_checked_views_native_equiv` binds that result to the actual
  FP32 Tensor words. All bit patterns, including NaNs, infinities and signed
  zeros, remain admitted. No finiteness or software-conformance law is needed.

`InputBinding.actual_pointers` retains the pointers observed by Python.
`TensorBridge.native_index_exact`, `native_mask_exact` and `checked_native_address`
reuse the existing local signed32/chunk proof and connect active native
addresses to these same Tensor views. Masked pointer formation does not require
backed storage. The existing chunk bound is a launch condition, not a new
total-input restriction.

The prior host theorems are preserved under `Misc/HostCorrect.lean` and
`Misc/HostFPEquiv.lean`. Neither public entry imports them. Historical host autotune, context and chunk-compatibility proofs remain optional; this revision
does not expand wrapper, compiler, runtime or allocator verification.

`Proofs/AccessSafety.lean` checks the actual structural execution of all 19
source statements. For every numerical backend, the three loads and two stores
reduce to the active P/G/M/P/M bounds. This is a regional access theorem; it
does not assert flat-pointer relocation, byte alignment or race freedom.
`Proofs/Interface.lean` composes that safety result with the existing execution
and frame proofs. The mathematical entry contains the IO specification, with statement-surface
and axiom audits; FPEquiv supplies the complete-grid program-pair specification. `TensorBridge` retains the prior full Tensor
statements through the same operational source lemmas used by the IO proofs.

## Native precision and numerical classification

The compiler-observed recurrence retains the actual FMA contraction and order:

```text
decay = fma(neg(lr), wd, 1)
difference = sub(m, g)
update = fma(beta1, difference, g)
chosen = gt(update, +0) ? sub(+0, lr) : lr
gate = neu(update, +0) ? 1 : +0
p' = fma(decay, p, mul(chosen, gate))
m' = fma(beta2, difference, g)
```

`Native.reference` is independent of the source graph evaluator.
`NativeWords` fixes +0 and 1 to `0x00000000` and `0x3f800000`. Its `neg`, `sub`,
`mul`, `fma`, unordered `neu` and ordered `gt` ports describe the native model
boundary. Negation and `sub(+0,lr)` remain distinct. No FMA/split identity,
reassociation or all-NaN-payload equality is assumed.

These six primitive ports are **not per-task numerical experiment TODOs**.
The historical `Words.LocalExact`/`NumericDomain` software bridge remains an
optional primitive-model helper. Its finite-site conditions do not narrow the
public theorem and do not become Python assertions.

`Native.lowering` and `Native.commuted_lowering` prove that the two final
momentum expressions lower to the same `fma(beta2, difference, g)` call, with
identical argument order. The parameter graph is unchanged. This is a strict
structural fact of the declared source-selected native interpretation, for
arbitrary instruction ports; it uses neither floating add commutativity nor a
split-multiply/add-to-FMA equality. The empty experiment rule set records no
numerical deferral. The selected native interpretation does not assert a
universal hardware/compiler conformance theorem. The legacy finite ADD-COMMUTE
proof is not used. No performance improvement is claimed.

## Focused validation

```sh
python bench/repairs/tritonbench_g/adam_update_triton/check_contract.py
python bench/repairs/tritonbench_g/adam_update_triton/check_cuda_lowering.py \
  --output-dir /tmp/adam-update-kernel-lowering
```

The CPU check preserves original/repair hashes and the kernel/autotune and
scalar-helper ASTs, checks all 19 DSL statements and all tensor guards, passes
34 rejection cases and 15 direct in-place cases, and reproduces the original
layout defect. Existing autotune restore, chunk boundary, empty and scalar
error checks are retained. CUDA administration is mocked.

Offline Triton 3.7.1 checks SM80/SM90, B128/B1024, and a divisible-by-16 mask
extent specialization. The observed graph retains FP32 runtime scalars, native
FMA, comparison behavior, and predicated memory; no explicit FTZ is emitted.
The checker follows the emitted word expressions into both stores and compares
their operation order and operand wiring with the independent recurrence.
These finite compiler observations are not a compiler proof or GPU execution.
No independent numerical replay or fresh GPU run is claimed.

Public proofs pass standard-base axiom audits and independent-formula/graph
noncircularity checks. The delivery packet records exact current file hashes
and targeted build, CPU and compiler evidence.

`Misc/TensorBridge.lean` contains the ordinary checked-Tensor applications, pointer/index helpers and `TensorBridge.python_checks_iff`. `Correct.lean` and `Misc/NativeCorrect.lean` retain the independent mathematical and native kernel results and do not import the bridge. `FPEquiv.lean` contains the complete-grid program-pair specification; optional single-source native results require an explicit Misc import.

## Program-pair completion, 2026-10-21

`Proofs/CommutedTile.lean` proves execution, all output cells and the full frame
for the actual candidate quote. `CommutedGrid.lean` lifts this to every complete
grid permutation. `WholePair.lean` connects both source runners to actual
`ProgramIO` invocation schedules, proves access safety along those executions,
and derives each entire output array from entry words. The final equivalence
uses the shared `ProgramIO.Implementation` syntax and a proved whole-program
structural step; no whole-output correspondence is a premise.

The runtime scalar ABI remains the successful result of the existing separate
integer-to-binary64 and binary64-to-FP32 conversions in `Proofs/Scalar.lean`.
Quantification over all four FP32 scalar words includes every successful ABI
binding without restricting finite values. Empty/error host paths and optional
host/Tensor theorems keep their existing scope. No new host, hardware or GPU
validation requirement is added.

## Optional proofs in Misc, 2026-10-21

Optional native observations, Tensor applications and reader/audit entries are
imported explicitly from `Misc/`. The primary `Contract.lean`, `Correct.lean` and
`FPEquiv.lean` keep the same mathematical and program-pair scope. Main proofs do
not re-export optional native conclusions. Historical optional declaration names and statements are preserved. The main
pair now proves the existing candidate at the same all-word native model and
precision, with no experimental rules.
