# Source and semantic boundary map

The selected source is `fused_rwkv6_kernel_scalar_abi_checked.py`, function
`fused_recurrent_rwkv6_fwd_kernel`. `Kernels.kernel` quotes the complete function
through an ordinary typed `triton` block and the shared `ComputeKernel`
evaluator. `tl.param scale : tl.float32` declares the existing launch scalar
register. The complete original function is definitionally equal to its former
macro-expanded source. The actual comparison candidate is the complete
`fused_rwkv6_kernel_u_early.py` module and `Kernels.earlyUKernel`.

| Python kernel operations | Quotation / execution proof |
| --- | --- |
| program IDs, head remainder, all six pointer setups, masks | `Kernels.kernel`; `Execution.entry` and `Context.entry_context` |
| FP32 zero tile and optional `0 + load(H0).to(float32)` | `Kernels.kernel`; `Context.entry_context`; independent `scalarSeed` |
| hoisted U load | `Execution.entry`; `Context.bias` |
| dynamic time loop and both traversal directions | `BodyExec.bodyExec`, `Body.body_context`, `Loop.loopExec` |
| masked K/V/Q/W loads, Q-scale product, exp, outer product | shared structural load/cast/binary/unary primitives; `Body.body_context` |
| pre-update output expression, axis-1 reduction, state update | `Context.terms`, `Context.carry`; `ModelBridge.native_rows` / `real_rows` |
| per-iteration scratch store before pointer advance | `Body.body_context`, `ScatterBridge.row_model` |
| five forward/reverse pointer advances | `Context` pointer invariant; `Execution.pointer_next` |
| optional final-state pointer and store | `Output.finishExec`, `Output.finish_memory` |
| early U pointer/mask/load, unchanged loop and complete schedule | `Pair.prefix_run`, `Pair.prefix_safe`, `Pair.kernel_run`, `Pair.kernel_safe`, `Pair.equivalent` |
| whole source and finite grid | `Output.source_memory`, `Grid.grid_model`, `Observations.Source.launch_memory` |
| actual Tensor data-pointer observations and four-byte stores | `Observations.tensorWord`, `Bytes.trace_realizes`, `FPEquiv.rwkv_native` |

The independent mathematical target is `Real.partialFormula` plus `realState`.
The native target is `Reference.native`: it reads the original input arrays and
uses the independent padded scalar recurrence, with no source execution or
computed output as input. Generic internal reduction-interface lemmas are
instantiated by the actual Real/native backends before the public endpoints.
`Correct.rwkv_real` retains the original `Sizes` domain; `FPEquiv.rwkv_native`
also exposes its primitive, compiler, reduction and scalar-binding ports. The
optional Tensor applications retain their physical guards, descriptors and
allocation/address facts. None takes successful execution, final memory equality
or an output-relation premise.

The quotation's natural pointer channel projects source signed offsets.
`Execution.pointer_signed_live` proves equality to the signed source address at
every dereference, including padded lanes. At reverse T=0 setup or the final
backedge a negative offset can be dead. `Loop.stopped_range` proves there is no
next body read, and `Output.finish_noninterference` proves final memory ignores
all five advanced pointer registers. The existing checked setup-width and
physical-address-room results remain in `fused_rwkv6_kernel/Proofs/`.

The source stores FP32 throughout, including explicit load and store casts.
Native graphs retain the seed addition, padded reduction leaves, multiplication
operand order and local FMA/contraction choices. Exp lowering uses the retained
`0x3fb8aa3b` multiplier followed by the exp2 approximation instruction. Raw scalar
ABI conversion is an input semantic port; there is no float coercion in the
Python repair. Raw integer 1 is a constexpr specialization and folds only its
designated scalar multiplication leaf in `Native.specialize`; float 1.0 remains
a runtime FP32 operand. The separate two-case installed-specializer receipt
pins this distinction without rebuilding prior compilation evidence.

Tensor contiguity/view interpretation, scalar binding, local primitive/ISA
meanings, structural reduction selection and compiler correspondence are model
boundaries. There are no registered numerical experiment assumptions. Runtime
grid-product compatibility remains the shared deferred follow-up. The new
proofs concern only this forward kernel's scratch partials, optional terminal
state and frame; Torch `sum(0)`, allocation execution, autograd and wrapper
execution are outside their scope.

The early-U transformation moves the same `p_u`, `mask_bk` and `b_u` statements
immediately after `i_h`, before the other pointer setup and optional H0 read.
This scheduling change has a latency-overlap purpose; no measured speedup is
claimed. Every numerical primitive, source precision, cast, padded reduction,
loop body, pointer advance and store remains unchanged. `FPEquiv.u_early_equiv`
compares actual original and candidate invocations under the empty experimental
rule set. Its structural proof covers all native payloads, both directions,
optional initial/final states, the zero-trip/no-launch case, the entire original
grid, safe successful executions, typed outputs and the complete exterior frame.
The native scalar ABI, compiler choices and reduction plan retain their original
explicit software-model meanings.
