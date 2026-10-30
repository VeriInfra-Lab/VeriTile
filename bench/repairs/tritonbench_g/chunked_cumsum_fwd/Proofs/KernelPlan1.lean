import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelExecution

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.NativePlan1
open VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel
set_option maxHeartbeats 20000000
set_option maxRecDepth 100000
-- Ordered native row graph SHA256 c281d17df5a74cfba81b7a9383932c4bf45ec98db0944707340a69c8652fd4e6
def node0 : SumTree 4 := .leaf ⟨0,by decide⟩
def node1 : SumTree 4 := .leaf ⟨1,by decide⟩
def node2 : SumTree 4 := .leaf ⟨2,by decide⟩
def node3 : SumTree 4 := .fused ⟨1,by decide⟩ node0
def node4 : SumTree 4 := .fused ⟨2,by decide⟩ node1
def node5 : SumTree 4 := .fused ⟨3,by decide⟩ node2
def node6 : SumTree 4 := .add node4 node0
def node7 : SumTree 4 := .add node5 node3

def row : Fin 4 → SumTree 4 := ![node0,node3,node6,node7]
theorem prefix_correct : ∀ j : Fin 4, (row j).leaves.Perm (List.range (j.val+1)) := by decide
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.NativePlan1
