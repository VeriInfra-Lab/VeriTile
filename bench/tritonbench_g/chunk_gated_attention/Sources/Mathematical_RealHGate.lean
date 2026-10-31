import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealHAlgebra

/-! Execution of each complete six-statement gate branch, including both
pointer constructions and both loads. Arbitrary tail values remain explicit;
these proofs impose no zero-padding premise. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHGate
open VeriTile.Triton RealHMath RealHProgram RealHPointers RealHTiles RealHLoads RealHAlgebra
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

structure Result (p : Layout) (x : Loaded p) (gateK : Bool) (f : StateValue p)
    (u v : BlockState) : Prop where
  carry : v.regs .real [p.BK,p.BV] "b_h" = some (stateTile p (scaled x gateK f))
  key : v.regs .real [p.BK,p.BT] "b_k" = some (Tile.mat (fun e t => some (gatedKey x gateK e t)))
  value : v.regs .real [p.BT,p.BV] "b_v" = some (Tile.mat (fun t j => some (gatedValue x gateK t j)))
  keyIndex : v.regs .nat [] "i_k" = u.regs .nat [] "i_k"
  valueIndex : v.regs .nat [] "i_v" = u.regs .nat [] "i_v"
  headIndex : v.regs .nat [] "i_bh" = u.regs .nat [] "i_bh"
  timeIndex : v.regs .nat [] "i_t" = u.regs .nat [] "i_t"
  mem : v.mem = u.mem
  pids : v.pids = u.pids
  undef : v.undef = u.undef

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHGate
