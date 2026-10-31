import bench.tritonbench_g.chunk_gla_simple.Sources.FPValue

namespace VeriTile.Bench.Repairs.ChunkGlaSimple.FPValue
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Host
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

structure Context {α : Type} (c : Shape) (initial current : State α) : Prop where
  memory : current.mem=initial.mem
  bh : current.regs .nat [] "i_bh"=some (fun _ => initial.pids 2)
  it : current.regs .nat [] "i_t"=some (fun _ => initial.pids 1)
  rows : current.regs .nat [c.BT] "rows"=some (fun i => row c initial i.1)
  cols : current.regs .nat [c.BV] "cols"=some (fun i => column c initial i.1)
  mask : current.regs .bool [c.BT,c.BT] "m_s"=some (fun i => decide (i.2.1.val ≤ i.1.val))

def HasAcc {α : Type} (c : Shape) (acc : Acc α c) (s : State α) : Prop :=
  s.regs .real [c.BT,c.BV] "b_o"=some acc.1 ∧ s.regs .real [c.BT,c.BT] "b_s"=some acc.2

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaSimple.FPValue
