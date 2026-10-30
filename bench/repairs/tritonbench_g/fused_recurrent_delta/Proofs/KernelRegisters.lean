import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelProgram
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelRegisters
open VeriTile.Triton FP.Structural KernelProgram
noncomputable section
variable {α : Type} [Inhabited α]

def keyAddress (s : Shape) (bh t i : Nat) := bh*(s.T*s.K)+i+t*s.K
def valAddress (s : Shape) (bh iv BV t j : Nat) := bh*(s.T*s.V)+iv*BV+j+t*s.V
def betaAddress (s : Shape) (bh t : Nat) := bh*s.T+t
def stateAddress (s : Shape) (bh iv BV i j : Nat) := bh*s.K*s.V+i*s.V+iv*BV+j
def keyPtr (s : Shape) (r : RegionName) (bh t : Nat) : Values α .ptr [s.BK] :=
  fun i => (r,keyAddress s bh t i.1.val)
def valPtr (s : Shape) (r : RegionName) (bh iv BV t : Nat) : Values α .ptr [BV] :=
  fun j => (r,valAddress s bh iv BV t j.1.val)
def betaPtr (s : Shape) (bh t : Nat) : Values α .ptr [] :=
  fun _ => ("beta",betaAddress s bh t)
def keyMask (s : Shape) : Values α .bool [s.BK] := fun i => decide (i.1.val<s.K)
def valMask (s : Shape) (iv BV : Nat) : Values α .bool [BV] := fun j => decide (iv*BV+j.1.val<s.V)
abbrev ForwardTile (s : Shape) (α : Type) := TileIndex [s.FBV,s.BK] → α

@[simp] theorem masked_regs {ι : Type} (r : RegionName) (off : ι → Nat)
    (value : ι → Cell α) (active : ι → Bool) (xs : List ι) (u : State α) :
    (xs.foldl (fun v i => if active i then v.write r (off i) (value i) else v) u).regs=u.regs := by
  induction xs generalizing u with
  | nil => rfl
  | cons i xs ih =>
    simp only [List.foldl_cons,ih]
    split <;> rfl

@[simp] theorem masked_pids {ι : Type} (r : RegionName) (off : ι → Nat)
    (value : ι → Cell α) (active : ι → Bool) (xs : List ι) (u : State α) :
    (xs.foldl (fun v i => if active i then v.write r (off i) (value i) else v) u).pids=u.pids := by
  induction xs generalizing u with
  | nil => rfl
  | cons i xs ih =>
    simp only [List.foldl_cons,ih]
    split <;> rfl

structure Forward (s : Shape) (f : Flags) (bh iv t : Nat) (h : ForwardTile s α) (u : State α) : Prop where
  q : u.regs .ptr [s.BK] "p_q"=some (keyPtr s "q" bh t)
  k : u.regs .ptr [s.BK] "p_k"=some (keyPtr s "k" bh t)
  v : u.regs .ptr [s.FBV] "p_v"=some (valPtr s "v" bh iv s.FBV t)
  o : u.regs .ptr [s.FBV] "p_o"=some (valPtr s "o" bh iv s.FBV t)
  beta : if f.vectorBeta then
    u.regs .ptr [s.FBV] "p_beta"=some (valPtr s "beta" bh iv s.FBV t)
    else u.regs .ptr [] "p_beta"=some (betaPtr s bh t)
  maskKey : u.regs .bool [s.BK] "mask_bk"=some (keyMask s)
  maskValue : u.regs .bool [s.FBV] "mask_bv"=some (valMask s iv s.FBV)
  carry : u.regs .real [s.FBV,s.BK] "h"=some h

end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelRegisters
