import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.RawKernel
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelExtended
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelLogBackward
namespace VeriTile.Bench.Repairs.KsoftmaxGridYChecked.RawReal
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section

def scores (K D r : Nat) (causal : Bool) (mode : Forward.MaskMode)
    (Q b : Nat) (s : State (WithBot ℝ)) : Fin D → WithBot ℝ :=
  KernelExtended.scores K r causal mode
    (fun j => (s.mem "left" (Native.flat Q K b r j.val)).read .real)
    (fun j => (s.mem "right" (Forward.maskAddress mode b r K j.val)).read .real)

theorem forward_kernel_real_raw (B Q K D : Nat) (hD : 0<D) (_covered : K≤D)
    (log causal : Bool) (mode : Forward.MaskMode) (s : State (WithBot ℝ))
    (domain : ∀ (b : Fin B) (r : Fin Q),0<K → KernelExtended.valid (scores K D r causal mode Q b s)) :
    ∃ final,RawKernel.run Forward.TypedExecution.model (.forward log causal mode) B Q K D s=some final ∧
      (∀ (b : Fin B) (r : Fin Q) (i : Fin D),i.val<K →
        final.mem "dst" (Native.flat Q K b r i.val)=Cell.mk .real
          (KernelExtended.formula (scores K D r causal mode Q b s) log i)) ∧
      (∀ (region : RegionName) j,region≠"dst" ∨ B*Q*K≤j → final.mem region j=s.mem region j) := by
  obtain ⟨final,execution,values,frame⟩ := RawKernel.complete Forward.TypedExecution.model (.forward log causal mode) B Q K D hD s
  refine ⟨final,execution,?_,frame⟩
  intro b r i hi
  rw [values b r i hi]
  exact KernelExtended.forward_value K r hD log causal mode _ _ (domain b r (by omega)) i

theorem backward_kernel_real_raw (B Q K D : Nat) (hD : 0<D) (_covered : K≤D)
    (log causal : Bool) (s : State (WithBot ℝ)) (gs os : Nat → ℝ)
    (hg : ∀ (b : Fin B) (r : Fin Q) (j : Fin D),Forward.readActive K r causal j.val=Bool.true →
      (s.mem "left" (Native.flat Q K b r j.val)).read .real=some (gs (Native.flat Q K b r j.val)))
    (ho : ∀ (b : Fin B) (r : Fin Q) (j : Fin D),Forward.readActive K r causal j.val=Bool.true →
      (s.mem "right" (Native.flat Q K b r j.val)).read .real=some (os (Native.flat Q K b r j.val))) :
    ∃ final,RawKernel.run Forward.TypedExecution.model (.backward log causal) B Q K D s=some final ∧
      (∀ (b : Fin B) (r : Fin Q) (i : Fin D),i.val<K →
        final.mem "dst" (Native.flat Q K b r i.val)=Cell.mk .real
          (some (gradient K r log causal (fun j => gs (Native.flat Q K b r j.val))
            (fun j => os (Native.flat Q K b r j.val)) i))) ∧
      (∀ (region : RegionName) j,region≠"dst" ∨ B*Q*K≤j → final.mem region j=s.mem region j) := by
  obtain ⟨final,execution,values,frame⟩ := RawKernel.complete Forward.TypedExecution.model (.backward log causal) B Q K D hD s
  refine ⟨final,execution,?_,frame⟩
  intro b r i hi
  rw [values b r i hi]
  exact backwardValue_real K r log causal _ _ _ _ (hg b r) (ho b r) i

theorem log_backward_kernel_real_raw (B Q K D : Nat) (hD : 0<D) (_covered : K≤D)
    (causal : Bool) (s : State (WithBot ℝ)) (gs : Nat → ℝ)
    (hg : ∀ (b : Fin B) (r : Fin Q) (j : Fin D),Forward.readActive K r causal j.val=Bool.true →
      (s.mem "left" (Native.flat Q K b r j.val)).read .real=some (gs (Native.flat Q K b r j.val))) :
    ∃ final,RawKernel.run Forward.TypedExecution.model (.backward Bool.true causal) B Q K D s=some final ∧
      (∀ (b : Fin B) (r : Fin Q) (i : Fin D),i.val<K →
        final.mem "dst" (Native.flat Q K b r i.val)=Cell.mk .real
          (some (KernelLogBackward.formula K r causal (fun j => gs (Native.flat Q K b r j.val))
            (fun j => (s.mem "right" (Native.flat Q K b r j.val)).read .real) i))) ∧
      (∀ (region : RegionName) j,region≠"dst" ∨ B*Q*K≤j → final.mem region j=s.mem region j) := by
  obtain ⟨final,execution,values,frame⟩ := RawKernel.complete Forward.TypedExecution.model (.backward Bool.true causal) B Q K D hD s
  refine ⟨final,execution,?_,frame⟩
  intro b r i hi
  rw [values b r i hi]
  exact KernelLogBackward.source_value K r causal _ _ _ (hg b r) i

#axiomsClean forward_kernel_real_raw
#axiomsClean backward_kernel_real_raw
#axiomsClean log_backward_kernel_real_raw
end
end VeriTile.Bench.Repairs.KsoftmaxGridYChecked.RawReal
