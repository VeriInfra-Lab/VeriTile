import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.NativePublicIO
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.CarryPublicIO
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.PairIO
import VeriTile.Spec

/-! Two actual ordinary grids on this task's fixed typed memory interface.
Every CTA executes from the common caller state; its actual source result is
merged through disjoint geometric write ownership. The implementation data
contains invocations, a pure word observer and write geometry only. -/

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.WholePair
open VeriTile Triton FP.Structural Source
noncomputable section
set_option maxHeartbeats 4000000

namespace Implementation
variable {A : Target.Ops α} {C : ScalarABI.Conversions α}
  {arg : ScalarABI.Argument} {B : Nat} {c : Config}

theorem of_implements {left right : Implementation A C arg B c}
    (hl : left.toIO.ImplementsOn (IOLayout.nativeDomain B c)
      (Mathematical.native A C arg B c))
    (hr : right.toIO.ImplementsOn (IOLayout.nativeDomain B c)
      (Mathematical.native A C arg B c)) :
    Spec.Equiv left right ([] : Spec.Assumptions (Implementation A C arg B c)) := by
  refine Spec.FloatingPoint.ofStructural (lhs := left) (rhs := right)
    (structural := Equivalent) rfl rfl ?_
  intro vals domain initial inputs
  obtain ⟨l,lr,ls,lo,lf⟩ := hl vals domain initial inputs
  obtain ⟨r,rr,rs,ro,rf⟩ := hr vals domain initial inputs
  refine ⟨l,r,lr,rr,fun capacity bounds => ⟨ls capacity bounds,rs capacity bounds⟩,?_,lf,rf⟩
  intro o i active
  exact (lo o i active).trans (ro o i active).symm

#axiomsClean of_implements
end Implementation

theorem equivalent (A : Target.Ops α) (C : ScalarABI.Conversions α)
    (arg : ScalarABI.Argument) (B : Nat) (c : Config) (hD : 0<c.D) (hN : 0<c.N)
    (hinj : Function.Injective (IOLayout.output (B:=B) c)) :
    Spec.Equiv (original A C arg B c) (candidate A C arg B c)
      ([] : Spec.Assumptions (Implementation A C arg B c)) := by
  exact Implementation.of_implements
    (NativePublicIO.implements A C arg B c hD hN hinj)
    (CarryPublicIO.implements A C arg B c hD hN hinj)

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.WholePair
