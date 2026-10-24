import bench.repairs.tritonbench_g.geglu_tanh_triton.Kernels
import bench.repairs.tritonbench_g.geglu_tanh_triton.Proofs.RawKernel
import VeriTile.Triton.Float.ProfiledMemory

namespace VeriTile.Bench.Repairs.GegluTanhTriton.RawNative
open VeriTile.Triton FP.Structural FP.ProfiledMemory
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Physical Native
open scoped VeriTile.Spec
noncomputable section

/-- The actual candidate changes only a signed integer address product. -/
def assumptions : Spec.Assumptions FP.GuardedFragment := []
def original (backward : Bool) (count : Nat) : Program := ⟨source backward 128 count,[]⟩
def indexCommuted (backward : Bool) (count : Nat) : Program :=
  ⟨indexCommutedSource backward 128 count,[]⟩

/-- Both actual shared sources succeed and agree on every typed memory cell.
All FP primitives, including FMA/tanh graph interpretation, are unchanged.
No finite-word restriction or numerical transformation law is needed. -/
theorem geglu_index_commuted_equiv (backward : Bool) (count : Nat) :
    original backward count ≡[assumptions] indexCommuted backward count := by
  apply VeriTile.Spec.FloatingPoint.ofNumerical
  · rfl
  · cases backward <;> rfl
  intro α _ M D _ s _
  let result := Tile.execution (FP.ProfiledRewrite.engine M) backward 128 count s
  exact ⟨result.val,result.val,result.property,
    (Rewrite.execution_eq (FP.ProfiledRewrite.engine M) backward 128 count s).symm.trans result.property,rfl⟩


#print_fp_assumptions geglu_index_commuted_equiv
#axiomsClean geglu_index_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.GegluTanhTriton.RawNative
