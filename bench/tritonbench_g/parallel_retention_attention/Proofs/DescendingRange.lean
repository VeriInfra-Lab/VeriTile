import bench.tritonbench_g.parallel_attention.Proofs.DescendingRange
import bench.tritonbench_g.parallel_retention_attention.Proofs.BackwardBounds

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.DescendingRange
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
  (BackwardConfig descendingCount descendingStart)
open VeriTile.Bench.Repairs.ParallelAttentionKernel.DescendingRange
  (mirrored_active mirrored_index ceil_active)

def lower (c : BackwardConfig) : Int :=
  -(((c.T+c.BTS-1)/c.BTS*c.BTS : Nat) : Int)+(c.BTS : Int)
def upper (c : BackwardConfig) : Int :=
  (c.BTS : Int)-(((c.i_c+1)*c.BTL : Nat) : Int)

theorem source_active (c : BackwardConfig) (positive : 0<c.BTS) (n : Nat) :
    lower c+(n : Int)*(c.BTS : Int)<upper c ↔ n<descendingCount c :=
  mirrored_active _ _ _ _ positive

theorem source_index (c : BackwardConfig) (positive : 0<c.BTS) (n : Nat)
    (live : n<descendingCount c) :
    -(lower c+(n : Int)*(c.BTS : Int))=(descendingStart c n : Int) := by
  have reached : n*c.BTS<(c.T+c.BTS-1)/c.BTS*c.BTS := by
    have bound:=(ceil_active n _ c.BTS positive).mp live
    exact lt_of_lt_of_le bound (Nat.sub_le _ _)
  have block : n<(c.T+c.BTS-1)/c.BTS:= (Nat.mul_lt_mul_right positive).mp reached
  have last := Nat.mul_le_mul_right c.BTS (show n+1≤(c.T+c.BTS-1)/c.BTS by omega)
  rw [Nat.add_mul,Nat.one_mul] at last
  exact mirrored_index _ _ _ last

#axiomsClean source_active
#axiomsClean source_index
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.DescendingRange
