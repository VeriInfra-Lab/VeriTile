# ksoftmax_triton

forward_kernel_real_raw: softmax/log-softmax from logical-K partition with exact mask behavior. backward_kernel_real_raw and log_backward_kernel_real_raw: corresponding finite-K derivatives, causal zeroing and extended-value rules.

- [ksoftmax_triton_grid_y_checked.py](ksoftmax_triton_grid_y_checked.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Three required Real specs over the complete selected B×Q grid, with D only the padded channel width and outputs restricted to lanes<K. Real selectedRows skips an empty logical product. Native FP pair reverses the complete independent row schedule, including harmless empty-K invocations.

## Conditions

- 0<D and explicit covered:K≤D in every main Real/FP result.
- ForwardDomain: at least one finite effective score per nonempty row, including optional absent/QK/BK mask and causal reads.
- BackwardDomain false retains finite active gradients and saved values; extended log-backward domain true retains finite active gradients while permitting saved negative infinity.
- Original FP32 typed source, mask/address modes, full row schedule and output windows remain; Real domains are not native word restrictions.

Exact FP program pairs and numerical dependencies are recorded in the [ksoftmax_triton contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
