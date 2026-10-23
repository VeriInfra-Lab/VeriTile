# chunk_cumsum_kernel: inclusive prefixes and carry commutation

For each row and valid time `t`, the independent target in `Contract.lean` is
`sum(input[row,k], 0 <= k <= t)`. It depends only on the initial input, time
length and output coordinate. It has no reference to kernel registers, chunk
size, padding, execution success or a candidate output.

`TensorBridge.kernel_correct` constructs successful execution of the actual selected
FP32 source in its Real interpretation, proves every valid output equals the
logical Tensor prefix, and preserves every memory cell outside the output row.
`InputReadLaw` in `Proofs/KernelContract.lean` binds each logical input coordinate
to the current physical heap at the Tensor's actual `data_ptr`. The public
theorem connects that heap to the source state through `Storage.lift`;
`TensorBinding.logical_input_read` proves the coordinate readback. Input and
output regions are distinct. The mathematical theorem erases floating-point
rounding; the separate `TensorBridge.chunk_sum_left_equiv` compares actual typed
floating-point executions.

## Source and admitted inputs

The production wrapper and kernel are retained. The wrapper admits contiguous
rank-three strided CUDA FP32 inputs with no unresolved negative view and an
output request of `None` or `torch.float32`. Nonempty inputs require
`B*H < 2**31` and a four-byte-aligned actual data pointer. Empty tensors return
the `empty_like` allocation without a launch. Contiguous offset views and
unusual singleton strides remain admitted. No copy or new input guard is added.
`Proofs/KernelContract.lean` proves the checked input conditions and supplies
ordinary Tensor metadata bindings; the public kernel proofs do not import the
optional host pipeline.

The same generic `InputReadLaw` serves Real cells and typed FP32 payload cells.
It is an ordinary input representation law, with no allocation, execution,
desired output or finite-value premise. `Proofs/KernelRealResult.lean` retains the
underlying arbitrary-memory theorem.

Both loop endpoints are signed 64-bit, row IDs and lane offsets widen before
arithmetic, and masked pointers replace the original native block pointers.
The five original configurations are `(BT, warps) = (16,2), (32,4), (32,2),
(64,8), (64,4)`. The numerical sequence remains FP32 load conversion, forward
inclusive scan, output carry addition, chunk reduction, carry update and output
conversion/store.

Unspecified masked tail loads are retained. They cannot enter valid forward
prefixes. Only the last carry update can see an inactive tail, and that carry
is never used by a later output. The Real proof and paired NaN-poison controls
cover this case without changing the source load fallback.

The original row product overflows signed i32 at `T=65`, row `33,038,210`:
`2,147,483,650` wraps to `-2,147,483,646`, even though the row fits CUDA grid-x.
`NativeIndex.original_row_overflow` checks this witness. The original flat
addressing also fails for noncontiguous views; the retained checked contract
rejects those views. The native proofs derive index/pointer bounds from ordinary
storage capacity, with no arbitrary new element-count cap. The loop endpoint
witness records why widening only the lower endpoint was insufficient.

## Actual candidate and floating-point rule

`Kernels.lean` contains readable original and `sumLeftKernel` DSL sources only.
The appended Python JIT `chunk_global_cumsum_scalar_kernel_sum_left` matches the
candidate. The sole change is `b_z += b_zz` to `b_z = b_zz + b_z`; production
dispatch remains as before.

`TensorBridge.chunk_sum_left_equiv` constructs successful executions of both actual
sources from the same typed state bound to the current physical input heap,
proves equality of their complete memories, returns both exterior frames, and
proves both results retain every logical Tensor input value. The underlying
arbitrary-state result remains `KernelEquivalence.memory_equiv` in
`Proofs/KernelEquivalence.lean`. `Proofs/Rewrite.lean` applies the shared strictly
proved FP32 `ADD-COMMUTE` rule to the actual carry and reduction operands in
reached source states. Its finite-operand domain applies only when a subsequent
chunk reads the updated carry. The final dead carry needs no numerical rule or
finite-value assumption. This rewrite domain is separate from Python admission;
it introduces no wrapper rejection and no desired-output or whole-success
premise.

The public FP theorem uses `Models []` of the fixed strict FP32 theory and has
no experimental `Rules` parameter. The rule is proved for the concrete
nearest-even software profile without input or result flushing. The existing
carry domain and all source/backend bindings are retained.

The backend retains the actual configured scan, reduction, literal/load/store
conversions and unspecified-load oracle. Their primitive semantics, rounding,
FTZ behavior and compiler binding are model boundaries, not additional deferred
rewrite experiments. The sole numerical transformation reported here is
FP32 finite-domain addition commutation. No associativity, FP-to-Real equality
or equality between different tuning configurations is claimed.

The old complete host/index-reference statements remain available under
`Misc/OptionalHostCorrect.lean` and `Misc/OptionalHostFPEquiv.lean`. They are
optional stronger scope and are not dependencies of the kernel entrypoints.
The storage representation uses FP32 payload cells at native addresses; it
does not verify a raw-byte encoding, Python runtime, compiler or CUDA launcher.

## Validation

The preceding validation record compiled its 26 local Lean modules with Lean
4.29.0, including the then-current public dependency closure and the preserved
optional host endpoints. Public
axiom audits contain only `propext`, `Classical.choice` and `Quot.sound`; the
used numerical derivation audit reports only `add_commute`. Source identities
and independent-target audits pass.

After compiling the local Lean closure, run:

```sh
python bench/repairs/tritonbench_g/chunk_cumsum_kernel/check_contract.py
python bench/repairs/tritonbench_g/chunk_cumsum_kernel/check_sum_left.py
```

The retained checker passes 12 rejection cases, 10 admissions, 60 CPU
Triton-interpreter controls and five NaN-poison tail controls, plus the original
layout/overflow witnesses. The paired checker binds the candidate AST to the
single carry swap and checks exact FP32 payload equality of complete output
backings, input preservation and exterior sentinels across all five configs,
tails, multiple chunks and storage offsets. Host controls mock CUDA ownership
and device context over real CPU Tensor metadata. These controls are concrete
source evidence, not a universal proof of the numerical backend. CUDA is
unavailable; no GPU execution is claimed. The delivery manifest records the
final Lean audit and source hashes.

`IO.lean` owns the canonical Real and FP records. `Correct.prefixIO` is a thin
alias of the Real view, retaining the selected source's mathematical projection,
one full-row input/output channel, source windows, masks, and empty scratch. `Correct.memory_correct` uses shared
`RegionalImplements`; it proves actual trace safety and successful execution
of every chunk, the independent inclusive prefix, and the complete raw-cell
frame. The existing distinct-region premise remains visible.

`IO.lean` displays the original and actual sum-left source schedules with
FP32 channels and identical windows. The single `FPEquiv.memory_equiv`
specification compares `original ≡[[]] sumLeft` through the shared implementation
view of `ProgramIO.EquivalentOn`, proving both actual traces safe, both source
loops successful, equal output cells and complete frames. In-place source/output
aliasing, arbitrary words and arbitrary unspecified tail values remain admitted.

`CarryDomain.chunkValues`, `chunkSum` and `carry` are independent recurrences
with the actual FP32 cast, reduction and addition order. The numerical domain
checks only pairs used by another chunk. `CarryDomain.iff_inputs` proves this
logical condition equivalent to the old reached-site condition for the same
input binding; this is a proved interface conversion, not a new numerical law
or experimental assumption. All five configurations and zero-length rows remain.

`Proofs/RealSafety.lean` and `Proofs/SourceSafety.lean` check actual loads/stores
through the complete signed loop. The public adapters reuse the established
complete source proofs. Original operational Real and complete typed-memory
FP results remain ordinary helper theorems in `Proofs/KernelRealResult.lean`
and `Proofs/KernelEquivalence.lean`. `Misc/TensorBridge.lean` reuses those foundations
with the same empty experimental rule set, actual pointer/logical input laws,
and input-preservation observations. Neither primary file imports the bridge.

This format pass targeted the changed Lean files and their Tensor consumers;
it did not rerun the historical CPU controls or GPU/numerical experiments.
Kernel source, Python guards, precision and the carry domain are unchanged.
The strict-rule migration removes the experimental ADD-COMMUTE parameter from
both the selected FP entrypoint and its registered Tensor consumer.
Its 25-module local import closure builds successfully. The direct
`#print_fp_assumptions memory_equiv` audit reports no experimental assumptions
and identifies the shared strict ADD-COMMUTE certificate; the standard-base
axiom and independent-domain audits pass.

The 2026-10-21 pair-entry revision writes the existing complete implementations
as `original ≡[[]] candidate` in `FPEquiv.lean`. `IO.lean` exposes both actual
source schedules and the shared `ProgramIO.Implementation` view. The existing
`EquivalentOn`/`Equivalent` proof is reused in `Proofs/ProgramPair.lean`, so both
actual traces remain safe, both runs succeed, all active output cells agree and
both exterior frames remain conclusions. The original strict `Models []` premise and the independent carry domain remain
explicit; only subsequently used carry pairs require finiteness. The shared
strict FP32 ADD-COMMUTE certificate is reused, with no empirical rule.
Python, source precision, arithmetic order and existing numerical evidence are
unchanged; validation is a targeted Lean build, with no new GPU experiment.

Optional application proofs and readers are under `Misc/`; their declaration namespaces and theorem statements are unchanged. Required main proof helpers remain under `Proofs/`.
