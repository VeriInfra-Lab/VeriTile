import bench.tritonbench_g.kldiv_compute.Sources.Mathematical_Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.KldivCompute
open VeriTile.Triton TypedSource
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

theorem typed_projects (x y output : Storage) (v : Bool)
    (X Y O : RegionName) (N B : Nat) :
    (kernel x y output v X Y O N B).toAlgorithm? =
      .ok (kernel x y output v X Y O N B).toAlgKernel := by
  cases x <;> cases y <;> cases output <;> cases v <;> rfl

theorem typed_supported (x y output : Storage) (v : Bool)
    (X Y O : RegionName) (N B : Nat) :
    (kernel x y output v X Y O N B).eraseDType.toAlgorithm? =
      .ok (kernel x y output v X Y O N B).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, typed_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivCompute
