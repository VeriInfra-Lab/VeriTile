/-
Source-audit target: _softmax_backward
Source-audit Lean: backwardSource%
Source-audit args: gradientType=tl.float32, savedType=tl.float32, destinationType=tl.float32, version=original
-/
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardProfiles
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardSource

/-!
The selected source is the complete backward JIT with all LOG, CAUSAL and
IS_FP16 branches. The imported macro separately exposes gradient, saved-output
and destination storage and retains all six strides, both masks and every cast.
Its candidate only commutes multiplication in the gradient input address.

The selector declares the fp32 audit view. The separate source-profile checker
compares all 216 original storage triples. Formal source families, native
compiler admission and independent replay are distinct obligations.

The historical forward source and its plain-only arithmetic theorem remain
in Proofs/ForwardLegacy.lean. The new ForwardSource quotation retains all
forward flags; its full mathematical contract selects fp32 storage, and its
typed state-equivalence family names four concrete storage triples.
-/
