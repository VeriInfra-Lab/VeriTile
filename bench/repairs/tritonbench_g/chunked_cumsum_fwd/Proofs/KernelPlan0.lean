import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelExecution

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.NativePlan0
open VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel
set_option maxHeartbeats 20000000
set_option maxRecDepth 100000
-- Ordered native row graph SHA256 94746d5b003aedd56ee61fea341eeaab427792d26a34d6277cdaf5d58bf2422d
def node0 : SumTree 1 := .leaf ⟨0,by decide⟩

def row : Fin 1 → SumTree 1 := ![node0]
theorem prefix_correct : ∀ j : Fin 1, (row j).leaves.Perm (List.range (j.val+1)) := by decide
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.NativePlan0
