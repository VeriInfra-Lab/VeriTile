import bench.tritonbench_g.matrix_transpose.Sources.Mathematical_Kernels
import VeriTile.Triton
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.MatrixTranspose
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

noncomputable def mathKernel (commuted : Bool) : Source :=
  fun M Out sx sy ox oy m d => (typedKernel commuted M Out sx sy ox oy m d).eraseDType

theorem typed_projects (v : Bool) (M Out : RegionName)
    (sx sy ox oy m d : Nat) :
    (typedKernel v M Out sx sy ox oy m d).toAlgorithm? =
      .ok (typedKernel v M Out sx sy ox oy m d).toAlgKernel := by
  cases v <;> rfl

theorem typed_projection (v : Bool) (M Out : RegionName)
    (sx sy ox oy m d : Nat) :
    (typedKernel v M Out sx sy ox oy m d).eraseDType.toAlgKernel =
      (mathKernel v M Out sx sy ox oy m d).toAlgKernel := rfl

theorem typed_supported (v : Bool) (M Out : RegionName)
    (sx sy ox oy m d : Nat) :
    (typedKernel v M Out sx sy ox oy m d).eraseDType.toAlgorithm? =
      .ok (typedKernel v M Out sx sy ox oy m d).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, typed_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

/-- Pointer aliases carry exactly the wrapper's fp16 memory reads. -/
theorem load_precision (v : Bool) (M Out : RegionName) (sx sy ox oy m d : Nat) :
    (typedKernel v M Out sx sy ox oy m d).surfaceBody[4]? =
      some (.assign .fp16 [m, d] "matrix" (.alg
        (.load .fp16 (.ptr (.ref .ptr [m, d] "matrix_ptr")) .none))) := by
  cases v <;> rfl

/-- The stored value is the actual rank-two transpose, in fp16 with no casts. -/
theorem transpose_store_precision (v : Bool) (M Out : RegionName)
    (sx sy ox oy m d : Nat) :
    (typedKernel v M Out sx sy ox oy m d).surfaceBody[5]? =
      some (.store .fp16 [d, m] (.ptr (.ref .ptr [d, m] "out_ptr"))
        (.alg (Op.transpose (batch := []) (.ref .fp16 [m, d] "matrix"))) .none) := by
  cases v <;> rfl

/-- Explicit input/output metadata resolves the source's pointer aliases. -/
theorem io_signature (v : Bool) (M Out : RegionName) (sx sy ox oy m d : Nat) :
    (typedKernel v M Out sx sy ox oy m d).inputs = [M] ∧
      (typedKernel v M Out sx sy ox oy m d).outputs = [Out] := by
  cases v <;> exact ⟨rfl, rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MatrixTranspose
