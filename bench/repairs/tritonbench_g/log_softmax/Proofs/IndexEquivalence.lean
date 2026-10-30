import bench.repairs.tritonbench_g.log_softmax.Kernels
import bench.repairs.tritonbench_g.log_softmax.Proofs.KernelGrid
import bench.repairs.tritonbench_g.log_softmax.Proofs.Contract
import bench.repairs.tritonbench_g.log_softmax.Proofs.NativeGrid
import VeriTile.Triton.Float.ProfiledMemory

namespace VeriTile.Bench.Repairs.LogSoftmax.IndexEquivalence
open VeriTile.Triton FP.Structural FP.ProfiledMemory
open scoped VeriTile.Spec
open Address
open VeriTile.Bench.Repairs.SoftmaxFlaggems
noncomputable section

/-- Only an integer row-index product changes. No numerical rule or finite-word
condition is required, and no primitive exp/log conformance experiment is due. -/
def assumptions : Spec.Assumptions FP.GuardedFragment := []

def forwardOriginal (O X : RegionName) (M N K BM C : Nat) : Program :=
  ⟨Source.forward O X M N K BM C,[]⟩
def forwardCommuted (O X : RegionName) (M N K BM C : Nat) : Program :=
  ⟨Source.forwardCommuted O X M N K BM C,[]⟩
def backwardOriginal (Y G O : RegionName) (M N K BM C : Nat) : Program :=
  ⟨Source.backward Y G O M N K BM C,[]⟩
def backwardCommuted (Y G O : RegionName) (M N K BM C : Nat) : Program :=
  ⟨Source.backwardCommuted Y G O M N K BM C,[]⟩

/-- Both complete actual forward sources succeed and agree on all typed memory
cells under arbitrary unchanged primitive, cast and reduction interpretations. -/
theorem forward_index_equiv (O X : RegionName) (M N K BM C : Nat) (hc : 0<C) :
    forwardOriginal O X M N K BM C ≡[assumptions] forwardCommuted O X M N K BM C := by
  apply Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro A _ algebra domain _ s _
  obtain ⟨t,ran⟩ := Kernel.forward_total (FP.ProfiledRewrite.engine algebra) O X M N K BM C s hc
  exact ⟨t,t,ran,(Rewrite.forward_profiled algebra O X M N K BM C s).symm.trans ran,rfl⟩

/-- The unchanged backward multiplication/subtraction, casts and sum retain
source order. Compiler FMA contraction remains the same lowering boundary for
both variants; no equality of differently rounded schedules is assumed. -/
theorem backward_index_equiv (Y G O : RegionName) (M N K BM C : Nat) :
    backwardOriginal Y G O M N K BM C ≡[assumptions] backwardCommuted Y G O M N K BM C := by
  apply Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro A _ algebra domain _ s _
  obtain ⟨t,ran⟩ := Kernel.backward_total (FP.ProfiledRewrite.engine algebra) Y G O M N K BM C s
  exact ⟨t,t,ran,(Rewrite.backward_profiled algebra Y G O M N K BM C s).symm.trans ran,rfl⟩


end
end VeriTile.Bench.Repairs.LogSoftmax.IndexEquivalence
