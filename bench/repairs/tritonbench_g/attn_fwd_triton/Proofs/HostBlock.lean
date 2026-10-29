import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.NativeLaunch
import Mathlib.Data.Nat.Log
namespace VeriTile.Bench.Repairs.AttnFwdTriton.HostShape
def blockD (D : Nat) := max 16 (2^Nat.clog 2 D)
theorem blockD_covers (D : Nat) : D≤blockD D :=
  le_trans (Nat.le_pow_clog (by decide) D) (Nat.le_max_right _ _)

open VeriTile.Triton

structure OutputDescriptor where
  region : RegionName
  base : Nat
  batches : Nat
  heads : Nat
  context : Nat
  headDim : Nat
  batchStride : Nat
  headStride : Nat
  rowStride : Nat
  dimStride : Nat
  dtype : TileDType

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.AttnFwdTriton.HostShape
