import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealHGate
import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealHStore

/-! One complete H iteration: construct pointers, store the old H, load K/V,
execute the selected gate branch, and update H. The execution and output
properties retain arbitrary unspecified lane values. Input separation is
required only against the history store performed before these loads. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHStep
open VeriTile.Triton RealHMath RealHProgram RealHPointers RealHTiles RealHLoads RealHAlgebra
noncomputable section
set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def historyBase (p : Layout) (l : Launch) (c : Nat) : Nat := l.head*p.historyHead+c*p.K*p.V

structure Separate (r : Regions) : Prop where
  key : r.key ≠ r.history
  value : r.value ≠ r.history
  gate : r.gate ≠ r.history

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHStep
