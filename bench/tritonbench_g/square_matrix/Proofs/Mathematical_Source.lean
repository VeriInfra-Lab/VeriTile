import bench.tritonbench_g.square_matrix.Sources.Mathematical_Kernels
import VeriTile.Triton
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SquareMatrix
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

noncomputable def mathKernel (commuted : Bool) : Source :=
  fun out inp si so n B => (typedKernel commuted out inp si so n B).eraseDType

theorem typed_projects (v : Bool) (out inp : RegionName)
    (si so n B : Nat) :
    (typedKernel v out inp si so n B).toAlgorithm? =
      .ok (typedKernel v out inp si so n B).toAlgKernel := by
  cases v <;> rfl

theorem typed_projection (v : Bool) (out inp : RegionName)
    (si so n B : Nat) :
    (typedKernel v out inp si so n B).eraseDType.toAlgKernel =
      (mathKernel v out inp si so n B).toAlgKernel := rfl

theorem typed_supported (v : Bool) (out inp : RegionName)
    (si so n B : Nat) :
    (typedKernel v out inp si so n B).eraseDType.toAlgorithm? =
      .ok (typedKernel v out inp si so n B).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, typed_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

/-- Alias-based pointers retain the actual input/output metadata. -/
theorem io_signature (v : Bool) (out inp : RegionName)
    (si so n B : Nat) :
    (typedKernel v out inp si so n B).inputs = [inp] ∧
      (typedKernel v out inp si so n B).outputs = [out] := by
  cases v <;> exact ⟨rfl, rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SquareMatrix
