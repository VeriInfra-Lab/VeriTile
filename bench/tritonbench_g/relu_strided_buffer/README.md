# ReLU on signed strided buffers: original conditional results

The original Python is unchanged. Historical whole-source statements are
restored as `Whole.relu_strided_buffer_real` in `Correct.lean` and
`Whole.relu_strided_buffer_fp32` in `FPEquiv.lean`, together with the existing
natural-stride slice and constant-folding declarations. Their historical
conditional domains and proofs remain intact.

The original successful domain is rank one, nonempty matching shapes, valid
signed views, nonzero output stride and disjoint accessed input/output
footprints. Both stride signs, input stride zero and disjoint subviews sharing
one allocation are included. The FP32 native-address statement additionally
requires `n < 2^31`, signed-int64 strides, one CUDA device, a successful device
guard and ordinary finite aligned allocation metadata. These premises are
conditions of the original theorem; original Python does not check them all.

The independent Real result is `max(0,x)`. The exact FP32 result for every word
is strict ordered positive selection: keep positive nonzero words, otherwise
store +0, including for NaNs and both zero signs. No numerical assumption is
needed. Theorems cover the 512-element tile cap, 65536-CTA cap, both source
branches, all output coordinates, returned caller `out0`, input/exterior cells
and bytes, actual signed pointer lowering and allocation bounds. Original empty
calls retain their separately proved tile-division failure before the CUDA guard.

The checked revision is documented in
[`bench/repairs/tritonbench_g/relu_strided_buffer`](../../repairs/tritonbench_g/relu_strided_buffer/README.md).
It checks the current domain before entering the unchanged wrapper body and
keeps the original StridedBuffer class, heuristics and kernels. Its source-only
`Kernels.lean` has a Lean identity bridge to this historical signed lowering.
The original `Kernels.lean` keeps its existing proof-dependent historical
source interface; proof cleanup does not change original acceptance scope.

The restored task-local `Proofs/` helpers retain historical names and bodies.
The old successful-call statements remain available independently of current
checked admission. These are source-semantics and ordinary runtime/Tensor API
results; no compiler verification or GPU hardware measurement is inferred.
