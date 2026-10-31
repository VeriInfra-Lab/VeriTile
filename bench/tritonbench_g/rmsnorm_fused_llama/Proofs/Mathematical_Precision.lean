import bench.tritonbench_g.rmsnorm_fused_llama.Sources.Mathematical_Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.Precision
noncomputable section
open VeriTile.Triton
set_option maxHeartbeats 16000000
set_option maxRecDepth 16000
set_option maxErrors 5
local instance : Inhabited ComputeStmt := ⟨.alg (.assign .nat [] "" (.constNat 0))⟩

def active (first : Bool) (N B : Nat) : Op .bool [B] :=
  if first then .lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N)
  else .ref .bool [B] "mask"

theorem io_roles (v : Bool)
    (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) :
    (typedKernel v X Y W stride N B eps).inputs = [X, W] ∧
    (typedKernel v X Y W stride N B eps).outputs = [Y] := by
  cases v <;> exact ⟨rfl, rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.Precision
