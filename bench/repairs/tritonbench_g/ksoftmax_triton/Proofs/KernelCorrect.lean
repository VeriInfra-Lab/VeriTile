import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelExtended
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelExecution
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelInputs

namespace VeriTile.Bench.Repairs.Ksoftmax.Correct
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section

def tensorForwardScores (D : Nat) (x : TensorMetadata) (mask : ArgumentMetadata) (mode : Option String)
    (causal : Bool) (xView : KernelTensor.View x)
    (maskView : KernelTensor.View (KernelInputs.maskTensor x mask mode))
    (projection : Graph.Word → WithBot ℝ) (memory : Nat → Graph.Word) (batch row : Nat) : Fin D → WithBot ℝ :=
  KernelExtended.scores (columns x) row causal (Host.decodeMode mode)
    (fun j => projection (KernelTensor.values memory x xView (Native.flat (rows x) (columns x) batch row j.val)))
    (fun j => projection (KernelTensor.values memory (KernelInputs.maskTensor x mask mode) maskView
      (Forward.maskAddress (Host.decodeMode mode) batch row (columns x) j.val)))

/-- Complete successful typed-source execution on the ordinary extended mask
domain, with logical Tensor inputs and every output/frame. The projection is
input-only. No finite-word guard or native/Real equality is assumed. -/
specification forward_kernel_real_extended (depth : Host.DepthLibrary) (y x : TensorMetadata)
    (mask : ArgumentMetadata) (log : Bool) (mode : Option String) (causal : Bool)
    (admitted : ForwardContract y x mask mode)
    (xView : KernelTensor.View x) (maskView : KernelTensor.View (KernelInputs.maskTensor x mask mode))
    (projection : Graph.Word → WithBot ℝ) (memory : Nat → Graph.Word) (s : State (WithBot ℝ))
    (domain : ∀ (b : Fin (batches x)) (r : Fin (rows x)), 0<columns x →
      KernelExtended.valid (tensorForwardScores (depth.depth (columns x)) x mask mode causal xView maskView projection memory b r)) :
    Spec.Real (∃ final,
      KernelExecution.exec Forward.TypedExecution.model depth (.forward log causal (Host.decodeMode mode)) x
        (KernelTensor.idealInitial projection y x (KernelInputs.maskTensor x mask mode) memory s)=some final ∧
      (∀ (b : Fin (batches x)) (r : Fin (rows x)) (i : Fin (depth.depth (columns x))), i.val<columns x →
        final.mem "dst" (Native.flat (rows x) (columns x) b r i.val)=Cell.mk .real
          (KernelExtended.formula
            (tensorForwardScores _ x mask mode causal xView maskView projection memory b r) log i)) ∧
      (∀ (region : RegionName) j, region≠"dst" ∨ x.numel≤j → final.mem region j=
        (KernelTensor.idealInitial projection y x (KernelInputs.maskTensor x mask mode) memory s).mem region j)) := by
  let initial := KernelTensor.idealInitial projection y x (KernelInputs.maskTensor x mask mode) memory s
  refine ⟨Host.body Forward.TypedExecution.model depth (.forward log causal (Host.decodeMode mode)) x initial,
    KernelExecution.success _ _ _ _ _,?_,?_⟩
  · intro b r i hi
    rw [Host.body_value _ _ _ _ admitted.2.2.1 initial b r i hi]
    have h := domain b r (by omega)
    simp only [tensorForwardScores,KernelTensor.values_words memory x xView admitted.2.1,
      KernelTensor.values_words memory (KernelInputs.maskTensor x mask mode) maskView
        (KernelInputs.mask_admitted y x mask mode admitted)] at h ⊢
    change forwardValue Forward.TypedExecution.model (columns x) r log causal (Host.decodeMode mode)
      (fun j => (initial.mem "left" (Native.flat (rows x) (columns x) b r j.val)).read .real)
      (fun j => (initial.mem "right" (Forward.maskAddress (Host.decodeMode mode) b r (columns x) j.val)).read .real) i=_
    simp only [initial,KernelTensor.ideal_left,KernelTensor.ideal_right]
    exact KernelExtended.forward_value _ _ (depth.positive _) _ _ _ _ _ h i
  · exact Host.body_frame _ _ _ _ admitted.2.2.1 initial

/-- Independent finite JVP bound to the actual input word projection. All
exceptional words remain covered separately by the native endpoint. -/
specification backward_kernel_real (depth : Host.DepthLibrary) (dst grad saved : TensorMetadata)
    (log causal : Bool) (admitted : BackwardContract dst grad saved)
    (gradView : KernelTensor.View grad) (savedView : KernelTensor.View saved)
    (projection : Graph.Word → WithBot ℝ) (memory : Nat → Graph.Word) (s : State (WithBot ℝ))
    (gs os : Nat → ℝ)
    (hg : ∀ (b : Fin (batches saved)) (r : Fin (rows saved)) (j : Fin (depth.depth (columns saved))),
      Forward.readActive (columns saved) r causal j.val=Bool.true →
      projection (KernelTensor.values memory grad gradView (Native.flat (rows saved) (columns saved) b r j.val))=
        some (gs (Native.flat (rows saved) (columns saved) b r j.val)))
    (ho : ∀ (b : Fin (batches saved)) (r : Fin (rows saved)) (j : Fin (depth.depth (columns saved))),
      Forward.readActive (columns saved) r causal j.val=Bool.true →
      projection (KernelTensor.values memory saved savedView (Native.flat (rows saved) (columns saved) b r j.val))=
        some (os (Native.flat (rows saved) (columns saved) b r j.val))) :
    Spec.Real (∃ final,
      KernelExecution.exec Forward.TypedExecution.model depth (.backward log causal) saved
        (KernelTensor.idealInitial projection dst grad saved memory s)=some final ∧
      (∀ (b : Fin (batches saved)) (r : Fin (rows saved)) (i : Fin (depth.depth (columns saved))), i.val<columns saved →
        final.mem "dst" (Native.flat (rows saved) (columns saved) b r i.val)=Cell.mk .real
          (some (gradient (columns saved) r log causal
            (fun j => gs (Native.flat (rows saved) (columns saved) b r j.val))
            (fun j => os (Native.flat (rows saved) (columns saved) b r j.val)) i))) ∧
      (∀ (region : RegionName) j, region≠"dst" ∨ saved.numel≤j → final.mem region j=
        (KernelTensor.idealInitial projection dst grad saved memory s).mem region j)) := by
  let initial := KernelTensor.idealInitial projection dst grad saved memory s
  refine ⟨Host.body Forward.TypedExecution.model depth (.backward log causal) saved initial,
    KernelExecution.success _ _ _ _ _,?_,?_⟩
  · intro b r i hi
    rw [Host.body_value _ _ _ _ admitted.2.2.2.1 initial b r i hi]
    apply backwardValue_real
    · intro j hj
      simp only [initial,KernelTensor.ideal_left]
      simpa only [KernelTensor.values_words memory grad gradView admitted.2.1] using hg b r j hj
    · intro j hj
      simp only [initial,KernelTensor.ideal_right]
      simpa only [KernelTensor.values_words memory saved savedView admitted.2.2.1] using ho b r j hj
  · exact Host.body_frame _ _ _ _ admitted.2.2.2.1 initial

#specNonCircular tensorForwardScores avoiding [forward,backward,runRows,Host.body,KernelExecution.exec]
#axiomsClean forward_kernel_real_extended
#axiomsClean backward_kernel_real
end
end VeriTile.Bench.Repairs.Ksoftmax.Correct
