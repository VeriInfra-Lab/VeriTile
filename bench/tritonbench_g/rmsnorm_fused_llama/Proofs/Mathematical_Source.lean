import bench.tritonbench_g.rmsnorm_fused_llama.Sources.Mathematical_Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RmsnormFusedLlama
noncomputable section
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- Exact Real projection of the repaired runtime call, including its scalar ABI. -/
def mathSource (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) : ComputeKernel :=
  (typedKernel Bool.false X Y W stride N B eps).eraseDType

def mathKernel (v : Bool) (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) : Kernel :=
  (typedKernel v X Y W stride N B eps).eraseDType.toAlgKernel

theorem typed_projects (v : Bool) (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) :
    (typedKernel v X Y W stride N B eps).toAlgorithm? =
      .ok (typedKernel v X Y W stride N B eps).toAlgKernel := by
  cases v <;> rfl

theorem typed_projection (v : Bool) (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) :
    (typedKernel v X Y W stride N B eps).eraseDType.toAlgKernel =
      mathKernel v X Y W stride N B eps := rfl

theorem typed_supported (v : Bool) (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) :
    (typedKernel v X Y W stride N B eps).eraseDType.toAlgorithm? =
      .ok (typedKernel v X Y W stride N B eps).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, typed_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.RmsnormFusedLlama
