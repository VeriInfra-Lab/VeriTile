import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelCorrect
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelIdeal

namespace VeriTile.Bench.Repairs.Ksoftmax.Correct
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section

/-- Mathematical input domain only; the Python guard still admits every word.
Unused causal/tail values and an ignored M have no finiteness restriction. -/
def ForwardWordDomain (depth : Host.DepthLibrary) (x : TensorMetadata) (mask : ArgumentMetadata)
    (mode : Option String) (causal : Bool) (xView : KernelTensor.View x)
    (maskView : KernelTensor.View (KernelInputs.maskTensor x mask mode)) (memory : Nat → Graph.Word) : Prop :=
  ∀ (b : Fin (batches x)) (r : Fin (rows x)) (i : Fin (depth.depth (columns x))),
    Forward.readActive (columns x) r causal i.val=Bool.true →
    KernelIdeal.finiteOrNegInf (KernelTensor.values memory x xView (Native.flat (rows x) (columns x) b r i)) ∧
    ((Host.decodeMode mode).present=Bool.true → KernelIdeal.finiteOrNegInf
      (KernelTensor.values memory (KernelInputs.maskTensor x mask mode) maskView
        (Forward.maskAddress (Host.decodeMode mode) b r (columns x) i)))

/-- Source-to-mathematics endpoint on actual decoded FP32 Tensor words.
Finite values use the existing exact rational decoder and -infinity stays
bottom. The independent partition must have a finite effective score. -/
specification forward_kernel_real_words (depth : Host.DepthLibrary) (y x : TensorMetadata)
    (mask : ArgumentMetadata) (log : Bool) (mode : Option String) (causal : Bool)
    (admitted : ForwardContract y x mask mode)
    (xView : KernelTensor.View x) (maskView : KernelTensor.View (KernelInputs.maskTensor x mask mode))
    (memory : Nat → Graph.Word) (s : State (WithBot ℝ))
    (_inputs : ForwardWordDomain depth x mask mode causal xView maskView memory)
    (domain : ∀ (b : Fin (batches x)) (r : Fin (rows x)), 0<columns x →
      KernelExtended.valid (tensorForwardScores (depth.depth (columns x)) x mask mode causal xView maskView
        KernelIdeal.decode memory b r)) :
    Spec.Real (∃ final,
      KernelExecution.exec Forward.TypedExecution.model depth (.forward log causal (Host.decodeMode mode)) x
        (KernelTensor.idealInitial KernelIdeal.decode y x (KernelInputs.maskTensor x mask mode) memory s)=some final ∧
      (∀ (b : Fin (batches x)) (r : Fin (rows x)) (i : Fin (depth.depth (columns x))), i.val<columns x →
        final.mem "dst" (Native.flat (rows x) (columns x) b r i.val)=Cell.mk .real
          (KernelExtended.formula
            (tensorForwardScores _ x mask mode causal xView maskView KernelIdeal.decode memory b r) log i)) ∧
      (∀ (region : RegionName) j, region≠"dst" ∨ x.numel≤j → final.mem region j=
        (KernelTensor.idealInitial KernelIdeal.decode y x (KernelInputs.maskTensor x mask mode) memory s).mem region j)) :=
  forward_kernel_real_extended depth y x mask log mode causal admitted xView maskView KernelIdeal.decode memory s domain

def BackwardWordDomain (depth : Host.DepthLibrary) (grad saved : TensorMetadata) (causal : Bool)
    (gradView : KernelTensor.View grad) (savedView : KernelTensor.View saved) (memory : Nat → Graph.Word) : Prop :=
  ∀ (b : Fin (batches saved)) (r : Fin (rows saved)) (i : Fin (depth.depth (columns saved))),
    Forward.readActive (columns saved) r causal i.val=Bool.true →
    KernelIdeal.finite (KernelTensor.values memory grad gradView (Native.flat (rows saved) (columns saved) b r i)) ∧
    KernelIdeal.finite (KernelTensor.values memory saved savedView (Native.flat (rows saved) (columns saved) b r i))

def tensorBackwardFormula (D : Nat) (grad saved : TensorMetadata) (log causal : Bool)
    (gradView : KernelTensor.View grad) (savedView : KernelTensor.View saved)
    (memory : Nat → Graph.Word) (batch row : Nat) (i : Fin D) : ℝ :=
  gradient (columns saved) row log causal
    (fun j => (KernelIdeal.decode (KernelTensor.values memory grad gradView
      (Native.flat (rows saved) (columns saved) batch row j))).getD 0)
    (fun j => (KernelIdeal.decode (KernelTensor.values memory saved savedView
      (Native.flat (rows saved) (columns saved) batch row j))).getD 0) i

specification backward_kernel_real_words (depth : Host.DepthLibrary) (dst grad saved : TensorMetadata)
    (log causal : Bool) (admitted : BackwardContract dst grad saved)
    (gradView : KernelTensor.View grad) (savedView : KernelTensor.View saved)
    (memory : Nat → Graph.Word) (s : State (WithBot ℝ))
    (domain : BackwardWordDomain depth grad saved causal gradView savedView memory) :
    Spec.Real (∃ final,
      KernelExecution.exec Forward.TypedExecution.model depth (.backward log causal) saved
        (KernelTensor.idealInitial KernelIdeal.decode dst grad saved memory s)=some final ∧
      (∀ (b : Fin (batches saved)) (r : Fin (rows saved)) (i : Fin (depth.depth (columns saved))), i.val<columns saved →
        final.mem "dst" (Native.flat (rows saved) (columns saved) b r i.val)=Cell.mk .real
          (some (tensorBackwardFormula _ grad saved log causal gradView savedView memory b r i))) ∧
      (∀ (region : RegionName) j, region≠"dst" ∨ saved.numel≤j → final.mem region j=
        (KernelTensor.idealInitial KernelIdeal.decode dst grad saved memory s).mem region j)) :=
  backward_kernel_real depth dst grad saved log causal admitted gradView savedView KernelIdeal.decode memory s
    (fun j => (KernelIdeal.decode (KernelTensor.values memory grad gradView j)).getD 0)
    (fun j => (KernelIdeal.decode (KernelTensor.values memory saved savedView j)).getD 0)
    (fun b r j h => KernelIdeal.decode_finite_value _ (domain b r j h).1)
    (fun b r j h => KernelIdeal.decode_finite_value _ (domain b r j h).2)

#specNonCircular ForwardWordDomain avoiding [forward,backward,runRows,Host.body,KernelExecution.exec]
#specNonCircular BackwardWordDomain avoiding [forward,backward,runRows,Host.body,KernelExecution.exec]
#specNonCircular tensorBackwardFormula avoiding [forward,backward,runRows,Host.body,KernelExecution.exec]
#axiomsClean forward_kernel_real_words
#axiomsClean backward_kernel_real_words
end
end VeriTile.Bench.Repairs.Ksoftmax.Correct
