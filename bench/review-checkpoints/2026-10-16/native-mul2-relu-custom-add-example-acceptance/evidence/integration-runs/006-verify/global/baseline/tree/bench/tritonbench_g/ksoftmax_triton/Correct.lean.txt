import bench.tritonbench_g.ksoftmax_triton.Kernels
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardPublic
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardPublic

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Correctness
open VeriTile Triton

/-- Complete backward original, with independent storage types and strides.
The explicit real erasure includes successful flat execution, the independent
finite-sum formula and the full frame outside active destination writes. -/
specification ksoftmax_backward_original_real_correct
    (gradient saved destination : Backward.Typed.Storage)
    (GradIn GradOut Out : RegionName) (bm bn gm gn om on K D : Nat)
    (log causal fp16 : Bool) :
    Spec.Real (Masked2DKernelIO₂.Implements
      (Backward.fullVariantIO gradient saved destination Bool.false GradIn GradOut Out
        bm bn gm gn om on K D log causal fp16)
      (fun _ row g o i => Backward.formula K row log causal g o i)) :=
  Backward.ksoftmax_backward_full_variant_real_correctness gradient saved destination
    Bool.false GradIn GradOut Out bm bn gm gn om on K D log causal fp16

/-- The complete candidate satisfies the same mathematical IO. Its proof
uses checked flat-execution transport, independently of typed FP equivalence. -/
specification ksoftmax_backward_candidate_real_correct
    (gradient saved destination : Backward.Typed.Storage)
    (GradIn GradOut Out : RegionName) (bm bn gm gn om on K D : Nat)
    (log causal fp16 : Bool) :
    Spec.Real (Masked2DKernelIO₂.Implements
      (Backward.fullVariantIO gradient saved destination Bool.true GradIn GradOut Out
        bm bn gm gn om on K D log causal fp16)
      (fun _ row g o i => Backward.formula K row log causal g o i)) :=
  Backward.ksoftmax_backward_full_variant_real_correctness gradient saved destination
    Bool.true GradIn GradOut Out bm bn gm gn om on K D log causal fp16

#axiomsClean ksoftmax_backward_original_real_correct
#axiomsClean ksoftmax_backward_candidate_real_correct
/-- Complete original fp32 forward source after explicit real erasure:
all source flags, three admitted mask modes, positive depth, independently
strided input/mask/output windows and the full unwritten-memory frame. -/
specification ksoftmax_forward_fp32_original_real_correct
    (Y X M : RegionName) (ym yn xm xn sm K D : Nat) (hD : 0 < D)
    (log : Bool) (mode : Forward.MaskMode) (causal fp16 : Bool) :
    Spec.Real (Masked2DKernelIO₂.Implements
      (Forward.fp32VariantIO Bool.false Y X M ym yn xm xn sm K D
        log mode causal fp16)
      (fun _ row xs masks i => Forward.formula K row log causal mode xs masks i)) :=
  Forward.ksoftmax_forward_fp32_variant_real_correctness Bool.false
    Y X M ym yn xm xn sm K D hD log mode causal fp16

/-- The complete fp32 candidate satisfies the same finite flat IO contract,
through independently proved equality of actual erased flat executions.
The typed floating equivalence supplies a separate guarantee. -/
specification ksoftmax_forward_fp32_candidate_real_correct
    (Y X M : RegionName) (ym yn xm xn sm K D : Nat) (hD : 0 < D)
    (log : Bool) (mode : Forward.MaskMode) (causal fp16 : Bool) :
    Spec.Real (Masked2DKernelIO₂.Implements
      (Forward.fp32VariantIO Bool.true Y X M ym yn xm xn sm K D
        log mode causal fp16)
      (fun _ row xs masks i => Forward.formula K row log causal mode xs masks i)) :=
  Forward.ksoftmax_forward_fp32_variant_real_correctness Bool.true
    Y X M ym yn xm xn sm K D hD log mode causal fp16

#axiomsClean ksoftmax_forward_fp32_original_real_correct
#axiomsClean ksoftmax_forward_fp32_candidate_real_correct

end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Correctness
