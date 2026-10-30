import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelRegisters
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelBackwardRegisters
open VeriTile.Triton FP.Structural KernelProgram KernelRegisters
noncomputable section
variable {α : Type} [Inhabited α]
abbrev Tile (s : Shape) (α : Type) := TileIndex [s.BK,s.BBV] → α
def plane (s : Shape) (bh iv : Nat) := bh+iv*s.B*s.H
def dbetaPtr (s : Shape) (bh iv t : Nat) : Values α .ptr [] :=
  fun _ => ("dbeta",plane s bh iv*s.T+t)

structure Reverse (s : Shape) (f : Flags) (bh iv t : Nat) (h : Tile s α) (u : State α) : Prop where
  q : u.regs .ptr [s.BK] "p_q"=some (keyPtr s "q" bh t)
  k : u.regs .ptr [s.BK] "p_k"=some (keyPtr s "k" bh t)
  v : u.regs .ptr [s.BBV] "p_v"=some (valPtr s "v" bh iv s.BBV t)
  do_ : u.regs .ptr [s.BBV] "p_do"=some (valPtr s "do" bh iv s.BBV t)
  dk : u.regs .ptr [s.BK] "p_dk"=some (keyPtr s "dk" (plane s bh iv) t)
  dv : u.regs .ptr [s.BBV] "p_dv"=some (valPtr s "dv" bh iv s.BBV t)
  beta : if f.vectorBeta then u.regs .ptr [s.BBV] "p_beta"=some (valPtr s "beta" bh iv s.BBV t)
    else u.regs .ptr [] "p_beta"=some (betaPtr s bh t)
  dbeta : if f.vectorBeta then u.regs .ptr [s.BBV] "p_dbeta"=some (valPtr s "dbeta" (plane s bh iv) iv s.BBV t)
    else u.regs .ptr [] "p_dbeta"=some (dbetaPtr s bh iv t)
  maskKey : u.regs .bool [s.BK] "mask_bk"=some (keyMask s)
  maskValue : u.regs .bool [s.BBV] "mask_bv"=some (valMask s iv s.BBV)
  carry : u.regs .real [s.BK,s.BBV] "d_h"=some h

structure Recompute (s : Shape) (f : Flags) (bh iv t : Nat) (h : Tile s α) (u : State α) : Prop where
  k : u.regs .ptr [s.BK] "p_k"=some (keyPtr s "k" bh t)
  v : u.regs .ptr [s.BBV] "p_v"=some (valPtr s "v" bh iv s.BBV t)
  do_ : u.regs .ptr [s.BBV] "p_do"=some (valPtr s "do" bh iv s.BBV t)
  dq : u.regs .ptr [s.BK] "p_dq"=some (keyPtr s "dq" (plane s bh iv) t)
  dk : u.regs .ptr [s.BK] "p_dk"=some (keyPtr s "dk" (plane s bh iv) t)
  dv : u.regs .ptr [s.BBV] "p_dv"=some (valPtr s "dv" bh iv s.BBV t)
  beta : if f.vectorBeta then u.regs .ptr [s.BBV] "p_beta"=some (valPtr s "beta" bh iv s.BBV t)
    else u.regs .ptr [] "p_beta"=some (betaPtr s bh t)
  maskKey : u.regs .bool [s.BK] "mask_bk"=some (keyMask s)
  maskValue : u.regs .bool [s.BBV] "mask_bv"=some (valMask s iv s.BBV)
  carry : u.regs .real [s.BK,s.BBV] "h"=some h

end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelBackwardRegisters
