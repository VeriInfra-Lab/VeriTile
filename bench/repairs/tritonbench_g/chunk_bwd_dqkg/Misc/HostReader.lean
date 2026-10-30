import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Misc.HostCorrect
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Domain
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Misc.HostFPEquiv
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.HostReader
open VeriTile Triton FP.Structural Whole
noncomputable section

/-- Direct checker-to-source Real application over independent Real inputs. -/
abbrev realApplication := @Direct.chunk_bwd_dqkg_real

/-- Reader application with the concrete staged algebra and dot model hidden
inside the public theorem. Only scalar/fixed-K8 agreements are parameters. -/
abbrev nativeApplication := @DirectNative.chunk_bwd_dqkg_if_local

#check realApplication
#check nativeApplication
#axiomsClean realApplication
#axiomsClean nativeApplication
#auditStmt Direct.chunk_bwd_dqkg_real
#auditStmt DirectNative.chunk_bwd_dqkg_if_local
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.HostReader
