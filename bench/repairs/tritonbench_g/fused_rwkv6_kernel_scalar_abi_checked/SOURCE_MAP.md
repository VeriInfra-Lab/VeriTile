# Source and semantic boundary map

Original: `fused_rwkv6_kernel_scalar_abi_checked.py`.
Candidate: `fused_rwkv6_kernel_u_early.py`.
Function: `fused_recurrent_rwkv6_fwd_kernel` in both files.

`Kernels.lean` and `CandidateKernels.lean` each contain five complete, directly
pasted bodies. Their `triton { ... }` content is checked against the corresponding
16-statement Python function body. Caller bindings live outside the block;
there are no extra `tl.region` or `tl.param` lines inside it.

| Caller scale | Original | Existing early-U candidate |
| --- | --- | --- |
| FP32, including float 1.0 | `kernel` | `earlyUKernel` |
| I32 | `kernelI32` | `earlyUKernelI32` |
| I64 | `kernelI64` | `earlyUKernelI64` |
| U64 | `kernelU64` | `earlyUKernelU64` |
| integer 1, native constexpr specialization | `kernelOne` | `earlyUKernelOne` |

The Python bodies are identical across caller profiles. `IO.Runtime.classify`
selects the actual dtype/specialization; the original `InputContract` implies
representability for each live launch. Raw integer arithmetic conversion remains
an operation in the interpreter, connected by `ScalarSource`/`ScalarTransport`.
`Native.scalarAlgebra` supplies the explicit ABI conversion primitive. It does
not erase a cast by assuming an arbitrary conversion is the identity.

| Python operations | Actual execution / safety connection |
| --- | --- |
| tuple program IDs, head remainder, pointer setups, masks | `Execution.entry`, `Context.entry_context` |
| zero tile and optional initial state | `Context.entry_context`, independent `scalarSeed` |
| U pointer, mask and load | `Context.bias`, `PairPrefix.prefix_run` / `prefix_safe` |
| signed `range(0,T)` and both directions | `BodyExec.bodyExec`, `Body.body_context`, `Loop.loopExec` |
| masked K/V/Q/W loads and casts, Q-scale product, exp, outer product | source primitives, `Body.body_context`, `ScalarSource` |
| pre-update output, axis-1 reduction, state recurrence | `Context.terms`, `ModelBridge.real_rows` / `native_rows`, `NativeScalar.scalar_rows` |
| per-iteration store and five signed pointer advances | `ScatterBridge.row_model`, `Execution.pointer_next` |
| optional final-state store | `Output.finishExec`, `Output.finish_memory` |
| each integer conversion through the full source | `ScalarTransport`, `ScalarSafety`, `RuntimeScalar` |
| disabled optional pointer arguments | `OptionalState.absent_channels_safe`, `RuntimeScalar.absent_channels_safe` |
| candidate prefix movement and complete suffix | `Pair`, `ScalarPair` |
| actual complete scalar-selected schedules | `RuntimeScalar.implements_iff`, `RuntimeScalar.equivalent` |
| independent Real mathematical result for all source profiles | `RuntimeReal.logical_real`, `Correct.rwkv_real_abi` |

The interpreter's natural pointer channel projects signed source offsets.
`Execution.pointer_signed_live` proves agreement at active dereferences.
Reverse T=0 setup and the final backedge may form negative dead offsets;
`Loop.stopped_range` rejects another iteration and `Output.finish_noninterference`
proves the exit ignores advanced pointer registers. The shared frontend retains
the signed increment before projecting the resulting pointer; it must not clip
a negative increment to zero before addition.

The query's nested load preserves its FP32 storage conversion, followed by the
explicit `.to(tl.float32)`. The independent ordered formula includes that event,
including the masked zero case. Native graphs retain the seed addition, padded
reduction leaves, multiplication order and local compiler choices. Integer 1
and floating 1.0 remain distinct in `Native.observeBound`; specialization is not
an IEEE multiplication identity.

`FPEquiv.u_early_equiv` compares actual original and candidate schedules under
an empty experimental rule set. It covers both traversal directions, optional
states, zero-trip/no-launch behavior, the full grid, safe successful execution,
typed output equality and the exterior cell frame. The independent Real target
is `Real.partialFormula` plus `realState`, never source execution or computed
output used as an input assumption.

Scalar binding, tensor layout, primitive/ISA meanings, compiler correspondence
and reduction-plan selection are explicit model boundaries. This kernel proof
does not claim Torch's final shard sum, allocation/autograd execution, runtime
scheduling compatibility, measured performance or a new numerical experiment.
