import bench.repairs.tritonbench_g.ksoftmax_triton.Kernels
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Host
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.ValueMath
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Counterexamples
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Pointers

namespace VeriTile.Bench.Repairs.Ksoftmax.Correct
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section

/-- Independent unshifted row softmax/log-softmax, including causal bottom. -/
def forwardFormula (Q K batch row D : Nat) (log causal : Bool) (mode : Forward.MaskMode)
    (xs masks : Nat → ℝ) (i : Fin D) : WithBot ℝ :=
  softmax (fun j => Forward.readActive K row causal j.val)
    (fun j => xs (Native.flat Q K batch row j.val)+
      if mode.present then masks (Forward.maskAddress mode batch row K j.val) else 0) log i

/-- Independent JVP; saved values are softmax or log-softmax coordinates. -/
def backwardFormula (Q K batch row D : Nat) (log causal : Bool) (gs os : Nat → ℝ) (i : Fin D) : ℝ :=
  gradient K row log causal (fun j => gs (Native.flat Q K batch row j.val))
    (fun j => os (Native.flat Q K batch row j.val)) i

specification forward_real (depth : Host.DepthLibrary) (y x : TensorMetadata)
    (mask : ArgumentMetadata) (log : Bool) (mode : Option String) (causal : Bool)
    (admitted : ForwardContract y x mask mode) (initial : State (WithBot ℝ)) (xs masks : Nat → ℝ)
    (hx : ∀ (b : Fin (batches x)) (r : Fin (rows x)) (j : Fin (depth.depth (columns x))),
      Forward.readActive (columns x) r causal j.val=Bool.true →
      (initial.mem "left" (Native.flat (rows x) (columns x) b r j.val)).read .real=some (xs (Native.flat (rows x) (columns x) b r j.val)))
    (hm : ∀ (b : Fin (batches x)) (r : Fin (rows x)) (j : Fin (depth.depth (columns x))),
      (Host.decodeMode mode).present=Bool.true → Forward.readActive (columns x) r causal j.val=Bool.true →
      (initial.mem "right" (Forward.maskAddress (Host.decodeMode mode) b r (columns x) j.val)).read .real=
        some (masks (Forward.maskAddress (Host.decodeMode mode) b r (columns x) j.val))) :
    Spec.Real (∃ final,
      Host.softmax Forward.TypedExecution.model depth y x mask log mode causal initial=.ok ((),final) ∧
      (∀ (b : Fin (batches x)) (r : Fin (rows x)) (i : Fin (depth.depth (columns x))), i.val<columns x →
        final.mem "dst" (Native.flat (rows x) (columns x) b r i.val)=Cell.mk .real
          (forwardFormula (rows x) (columns x) b r _ log causal (Host.decodeMode mode) xs masks i)) ∧
      (∀ (region : RegionName) j, region≠"dst" ∨ x.numel≤j → final.mem region j=initial.mem region j)) := by
  refine ⟨Host.body Forward.TypedExecution.model depth (.forward log causal (Host.decodeMode mode)) x initial,
    Host.forward_return _ _ _ _ _ _ _ _ admitted initial,?_,?_⟩
  · intro b r i hi
    rw [Host.body_value _ _ _ _ admitted.2.2.1 initial b r i hi]
    exact forwardValue_real (columns x) r (by omega) (depth.positive _) log causal (Host.decodeMode mode)
      _ _ _ _ (hx b r) (hm b r) i
  · exact Host.body_frame _ _ _ _ admitted.2.2.1 initial

specification backward_real (depth : Host.DepthLibrary) (dst grad saved : TensorMetadata)
    (log causal : Bool) (admitted : BackwardContract dst grad saved) (initial : State (WithBot ℝ)) (gs os : Nat → ℝ)
    (hg : ∀ (b : Fin (batches saved)) (r : Fin (rows saved)) (j : Fin (depth.depth (columns saved))),
      Forward.readActive (columns saved) r causal j.val=Bool.true →
      (initial.mem "left" (Native.flat (rows saved) (columns saved) b r j.val)).read .real=
        some (gs (Native.flat (rows saved) (columns saved) b r j.val)))
    (ho : ∀ (b : Fin (batches saved)) (r : Fin (rows saved)) (j : Fin (depth.depth (columns saved))),
      Forward.readActive (columns saved) r causal j.val=Bool.true →
      (initial.mem "right" (Native.flat (rows saved) (columns saved) b r j.val)).read .real=
        some (os (Native.flat (rows saved) (columns saved) b r j.val))) :
    Spec.Real (∃ final,
      Host.softmaxBackward Forward.TypedExecution.model depth dst grad saved log causal initial=.ok ((),final) ∧
      (∀ (b : Fin (batches saved)) (r : Fin (rows saved)) (i : Fin (depth.depth (columns saved))), i.val<columns saved →
        final.mem "dst" (Native.flat (rows saved) (columns saved) b r i.val)=Cell.mk .real
          (some (backwardFormula (rows saved) (columns saved) b r _ log causal gs os i))) ∧
      (∀ (region : RegionName) j, region≠"dst" ∨ saved.numel≤j → final.mem region j=initial.mem region j)) := by
  refine ⟨Host.body Forward.TypedExecution.model depth (.backward log causal) saved initial,
    Host.backward_return _ _ _ _ _ _ _ admitted initial,?_,?_⟩
  · intro b r i hi
    rw [Host.body_value _ _ _ _ admitted.2.2.2.1 initial b r i hi]
    exact backwardValue_real (columns saved) r log causal _ _ _ _ (hg b r) (ho b r) i
  · exact Host.body_frame _ _ _ _ admitted.2.2.2.1 initial

#specNonCircular forwardFormula avoiding [forward,backward,Host.body,Host.softmax]
#specNonCircular backwardFormula avoiding [forward,backward,Host.body,Host.softmaxBackward]
#axiomsClean forward_real
#axiomsClean backward_real
end
end VeriTile.Bench.Repairs.Ksoftmax.Correct
