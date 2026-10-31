import bench.tritonbench_g.token_attn_reduceV.Proofs.Mathematical_FPModel
namespace VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

structure Ready {α : Type} [Inhabited α] (p : Parameters) (s t : State α)
    (acc : TileIndex [p.width] → α) : Prop where
  mem : t.mem = s.mem
  pids : t.pids = s.pids
  undef : t.undef = s.undef
  batch : t.regs .nat [] "cur_batch" = some (fun _ => s.pids 0)
  head : t.regs .nat [] "cur_head" = some (fun _ => s.pids 1)
  offsN : t.regs .nat [p.block] "offs_n" = some (fun i => i.1.val)
  offsD : t.regs .nat [p.width] "offs_d" = some (fun i => i.1.val)
  length : t.regs .nat [] "cur_batch_seq_len" = some (fun _ => count p s)
  prob : t.regs .nat [p.block] "p_offs" = some (fun i => probOffset p s i.1.val)
  gather : t.regs .nat [p.block] "v_loc_off" = some (fun i => gatherOffset p s i.1.val)
  value : t.regs .nat [1,p.width] "v_offs" = some (fun i => valueOffset p s i.2.1.val)
  accumulator : t.regs .real [p.width] "acc" = some acc

end
end VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages
