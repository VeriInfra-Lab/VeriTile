# chunked_cumsum_fwd guarded FP32 repair

<!-- current-isolated-kernel-entry -->
## Current kernel-only revision

Accepted after complete consumer compilation, independent review and root import checks. Complete selected kernel quotation and grid; independent Real preparation/prefix formulas and native FP32 ordered-scan target; actual input binding, both Tensor outputs, common byte memory and exterior frame. Checked input domain and source precision retained.

The primary review entries are [KernelOnly/Correct.lean](KernelOnly/Correct.lean) and [KernelOnly/FPEquiv.lean](KernelOnly/FPEquiv.lean). The mathematical entry contains the independent `ProgramIO` preparation and inclusive-prefix specification. The FP entry compares the complete original and existing scale-first source programs. [KernelOnly/IO.lean](KernelOnly/IO.lean) declares the dt/rate/optional-bias arrays, both complete packed output arrays, actual strides, independent runtime clamp registers and the full source launch schedule. Input words, including all special values, remain unrestricted. Both judgments prove actual-source access safety, successful execution, typed outputs and the exact exterior cell frame. The prior single-source ordered-native theorem `FPEquiv.kernel_fp32` is retained separately in [KernelOnly/Misc/NativeCorrect.lean](KernelOnly/Misc/NativeCorrect.lean).

[KernelOnly/Misc/TensorBridge.lean](KernelOnly/Misc/TensorBridge.lean) preserves the full old `repaired_real` and `repaired_fp32` statements as ordinary theorems in `TensorBridge.Real` and `TensorBridge.Native`, using the same identified operational source lemmas as the IO proofs. Both actual Tensor outputs, shared physical memory, byte/input/exterior frames, access/alignment facts and scalar ABI conversion remain covered.

[verified-entry.json](verified-entry.json) retains the earlier frozen source/dependency generation. The 2026-10-21 implementation pair uses the additive shared `minimumFloat`/`Algebra.minimum` support and the isolated [minimum-support build driver](../../../../.veritile-cache/finish-pending-105-20261021/minimum-support/build.py). Its objects remain outside the shared strict overlay. Rebuilding this new pair therefore requires that driver until root integrates the matching shared source and affected object generation; the old strict overlay alone lacks the selected minimum primitive. The campaign delivery records the exact 14 shared paths, source preservation, module receipts and fresh joint check. Python, input contract and frozen files are unchanged.

From the repository root, rebuild the frozen proof closure with one command:

```sh
python3 .veritile-cache/task-deliveries/chunked_cumsum_fwd/authorized-minimum-v1/replay_generation.py .veritile-cache/task-deliveries/chunked_cumsum_fwd/authorized-minimum-v1/replay-plan.json
```

The current kernel-only recovery has no new numerical-experiment deferral.
The old deferred-family classification below belongs to the historical
contract/host review. The general theorem retains explicit scalar and scan-order boundaries; concrete
PTX scan evidence covers 42 saved SM80 profiles at widths 1, 4 and 64. This does
not assert GPU execution or compiler-profile coverage at every admitted width.

## Historical contract and host review

The retained text below describes the earlier review. Its public-entry paths, open items and numerical-deferral classification are historical; the current selection and exact proof/model boundaries are recorded above and in `verified-entry.json`.

The independent target first prepares each original logical `dt[b,t,h]`: add the optional head bias, optionally apply cutoff softplus (`log(1+exp(x))` when `x <= 20`, otherwise `x`), apply the lower then upper clamp, and write zero for padded time coordinates. `dt_out` contains these prepared values. `dA_cumsum` is their inclusive chunk-local prefix after multiplication by `A[h]`. Both outputs have shape `(B,H,ceil(S/chunk_size),chunk_size)` and FP32 storage. The wrapper allocates `dt_out` first but returns `(dA_cumsum, dt_out)`.

The original numerical kernel, its axis-1 scan and all seven autotune configurations (`BLOCK_SIZE_H = 1,2,4,8,16,32,64`) are retained. The repair computes the chunk count with exact integer ceiling and widens program IDs, lane coordinates and sequence/head/chunk arithmetic to uint64 before products. It passes the original strided Tensor objects directly, including nonzero offsets, noncontiguous inputs, zero strides and input aliases. The original benchmark Python is unchanged.

The admitted arguments are strided FP32 CUDA Tensors on the same device, with no lazy-negative bit and four-byte-aligned nonempty data pointers. `dt` has rank three; `A` and optional `dt_bias` have shape `(H,)`. `chunk_size` is a positive Python integer at most 1048576 (and checked below 2^63); Python bool follows `isinstance(..., int)`. `dt_softplus` is bool and `dt_limit` is a two-element tuple of Python floats. Every `dt` dimension is below 2^63. For nonempty `dt`, `B < 2^31`, `ceil(S/chunk_size) <= 65535` and `H <= 65535`, the last bound accommodating the configuration with one head per program. Empty shapes retain the original two allocations and zero-grid source launch. Device context follows `dt.device`.

`Contract.lean` names every condition. `checkInput_iff` proves the exact admitted Tensor domain after dynamic Tensor/None and scalar-type classification; `classifyTensor` rejects non-Tensors before querying attributes. Admission is about successful-domain agreement, not which error wins when multiple checks fail. The contract does not restrict finite values or clamp ordering. The Real entry is an independent mathematical projection with real inputs, a real lower clamp and a finite or positive-infinite upper clamp. The FP entry additionally covers arbitrary input, rate, bias and clamp words, including NaNs and infinities, without asserting equality to the Real result.

Read `Kernels.lean`, `Correct.lean`, then `FPEquiv.lean`. `Kernels.lean` contains only the source template; `Proofs/Source.lean` proves the template binding and defines the widened index projection. Existing valid source execution, complete grids, scan-tree, host ABI and storage proofs were reused. Helpers do not import the public readers. No historical compiler evidence bundle or acceptance metadata was copied.

`Correct.repaired_real` covers the guarded wrapper, the independent preparation/prefix formulas, allocation and setup failures, exact returned objects, metadata and caller memory frame. `FPEquiv.repaired_fp32` includes all output words, all four bytes of each FP32 result, the flat native-memory representation, every original caller byte including partial trailing words, and the output-exterior frame. Arbitrary autotune trial sequences thread their actual preceding heaps before the selected complete launch. `Prepared` retains the exact original input descriptors and allocation heap; it performs no input copy or negative-view resolution. `TensorBinding` connects checked shapes/devices to this call, and `PointerBinding` connects checked actual pointers to its backing descriptors. Original Tensor stride observations are retained in the call and backing interface.

Ordinary successful allocation supplies fresh output objects, live backing windows, storage extents and address separation. These contracts contain no desired kernel result. All dimension bounds used by the native masks are derived by `InputContract.geometrySizes`; they are not allocator premises. `InputContract.geometryArange` proves that the power-of-two block is at most 1048576, so source arange coordinates are representable before widening. The source's active byte addresses, allocation windows and uint64 mask comparisons are proved from these checks and ordinary same-call backing bounds. Masked lanes may have wrapped unused pointer arithmetic. The separate shared runtime grid-product compatibility issue remains deferred.

The FP reference retains the selected native prefix tree, explicitly marked FMA contractions and configured bias operand order. The preparation graph preserves the inspected `exp(x)+1` order, ordered max/min operations and eager computation before remasking. It assumes neither scan reassociation nor split-multiply/add equality with FMA. Runtime clamp words use the explicit x86-64 default-rounding binary64-to-FP32 ABI algorithm, including subnormals, infinities, signed zeros and quieted NaN payload conversion; conversion errors remain setup outcomes. There is no whole-output numerical callback.

Ten local FP32 numerical families remain deferred: literals 0/1/20; input/compute/store casts; ordered addition; multiplication; maximum; minimum; exponential; logarithm; cutoff comparison; and ternary FMA. Their domains are the exact reached scalar operands, including eager masked-lane computation and arbitrary FP32 special values. The source-specific prefix tree and operand-order selection are explicit compiler-boundary parameters. The remaining experiment is to pin the compiler/device profile, inspect the selected graphs across all seven configurations and relevant chunk widths, then replay those scalar sites against the intended rounding, FTZ, signed-zero and NaN policies. Restored finite scan-plan proofs are not fresh compiler or GPU evidence. No GPU execution, numerical conformance or FP-to-Real error bound is claimed.

Validation: the restored dependencies and 15 repair Lean modules compile with standard-base-only public axiom audits. `python bench/repairs/tritonbench_g/chunked_cumsum_fwd/check_contract.py` checks the unchanged complete numerical AST after erasing only the added integer casts, 20 rejected inputs, 36 admitted inputs and 936 program executions across all seven configurations. Controls cover optional bias/softplus combinations, strided and broadcast inputs, non-power-of-two chunks, tails, empty axes, original pointer identity, allocation/return order, fresh-output guard storage and untouched inputs. It also checks the original lazy-negative and floating-ceiling counterexamples. This is CPU backing-storage replay with mocked CUDA launch/allocation, not native scan or scalar conformance.

## Implementation-pair review (2026-10-21)

`KernelOnly.FPEquiv.chunked_cumsum_scaleFirst_equiv` now proves the actual
`original ≡[[]] scaleFirst` pair. [Kernels.lean](Kernels.lean) contains both
complete ordinary `triton` quotations, differing only at `dt * A[:, None]`
versus `A[:, None] * dt`. This reuses the preserved scale-first candidate to
expose its alternate multiplicand routing; no performance measurement is claimed.
The complete original quotation equals the selected previous runtime-bound
quotation by `rfl` under the restored minimum frontend.

The pair covers the same geometry and size conditions, all raw FP32 input and
clamp words, both optional branches, padding, all launches, both output arrays,
source safety and exterior frames. The native scan plan remains arbitrary.
Ordinary leaves use the shared strict `ScalarOpsLaws.mul_comm`; each explicit
fused leaf uses the shared strict `ScalarOpsLaws.fma_mul_comm`, preserving its
addend and tree position. No split-multiply/add to FMA rewrite, reassociation,
finite-input restriction or experimental numerical premise is introduced.
The other preparation/cast/comparison ports are unchanged explicit model data
on both programs. The empty rule set is relative to this fixed software model,
and does not assert universal hardware equivalence.

Validation uses the isolated minimum-support generation for all selected main
entries and the retained native entry. A representative completed nonminimum
`batched_vecmat_mult` ProgramIO pair is rebuilt against the same shared objects
and included in the fresh joint. This compatibility check does not certify every
other task's old strict-overlay object generation.

## Optional proof files (2026-10-21)

The optional native, Tensor, audit and application entries for this selected revision are in `KernelOnly/Misc`:

- `KernelOnly/Misc/NativeCorrect.lean`
- `KernelOnly/Misc/TensorBridge.lean`

The primary Contract/Correct/FPEquiv entries and their required IO, source and Proofs dependencies keep their existing statements and scope. This layout change adds no mathematical or numerical claims.
