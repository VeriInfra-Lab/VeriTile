import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardRemainingMath
import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardSecondPrimitives
import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardFirstState

/-! Reverse-loop register state. Pointer equations are needed only if a row
remains to be read; the terminal pointers keep their actual clipped subtraction
results. Numerical carry and gradients use independent initial-memory sums. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardSecondState
open VeriTile.Triton Primitives MaskedMath
open ForwardState (keyPointers valuePointers outputPointers)
open BackwardFirstState (keyGradientPointers)
noncomputable section
set_option maxHeartbeats 1200000
set_option linter.unusedSimpArgs false

def dStateTile (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (t : Nat) : Tile .real [p.BK,p.BV] :=
  matrix p.BK p.BV (dStateValue p l initial r.q r.do_ t)

def carryTile (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (remaining : Nat) : Tile .real [p.BK,p.BV] :=
  matrix p.BK p.BV (BackwardRemainingMath.carry p l initial r remaining)

structure Ready (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (remaining : Nat) (u : BlockState) : Prop where
  keyId : u.regs .nat [] "i_k" = some (Tile.scalar l.key)
  valueId : u.regs .nat [] "i_v" = some (Tile.scalar l.value)
  headId : u.regs .nat [] "i_bh" = some (Tile.scalar l.head)
  decay : u.regs .real [] "b_b" = some (Tile.scalar (some (MaskedMath.decay p l)))
  q : 0 < remaining → u.regs .ptr [p.BK] "p_q" = some (keyPointers p l r.q (remaining-1))
  k : 0 < remaining → u.regs .ptr [p.BK] "p_k" = some (keyPointers p l r.k (remaining-1))
  v : 0 < remaining → u.regs .ptr [p.BV] "p_v" = some (valuePointers p l r.v (remaining-1))
  dout : 0 < remaining → u.regs .ptr [p.BV] "p_do" = some (valuePointers p l r.do_ (remaining-1))
  dk : 0 < remaining → u.regs .ptr [p.BK] "p_dk" = some (keyGradientPointers p l r.dk (remaining-1))
  dv : 0 < remaining → u.regs .ptr [p.BV] "p_dv" = some (outputPointers p l r.dv (remaining-1))
  maskK : u.regs .bool [p.BK] "mask_bk" = some (maskTile p.BK (fun k => activeKey p l k.val))
  maskV : u.regs .bool [p.BV] "mask_bv" = some (maskTile p.BV (fun j => activeValue p l j.val))
  carry : u.regs .real [p.BK,p.BV] "d_h" = some (carryTile p l initial r remaining)

def beforeStores (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (t : Nat) (u : BlockState) : BlockState :=
  let u1 := u.setReg "_do" .real [p.BV] (vector p.BV (valueValue p l initial r.do_ t))
  let u2 := u1.setReg "_q" .real [p.BK] (vector p.BK (fun k => keyValue p l initial r.q t k*p.scale))
  let u3 := u2.setReg "_k" .real [p.BK] (vector p.BK (keyValue p l initial r.k t))
  let u4 := u3.setReg "_v" .real [p.BV] (vector p.BV (valueValue p l initial r.v t))
  let u5 := u4.setReg "d_h" .real [p.BK,p.BV] (dStateTile p l initial r t)
  let u6 := u5.setReg "d_k" .real [p.BK] (vector p.BK (MaskedMath.dk p l initial r t))
  let u7 := u6.setReg "d_v" .real [p.BV] (vector p.BV (MaskedMath.dv p l initial r t))
  u7.setReg "d_h" .real [p.BK,p.BV] (carryTile p l initial r t)

def scatterK (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (t : Nat) (u : BlockState) : BlockState :=
  (TileShape.allIndices [p.BK]).foldl (fun s k => if activeKey p l k.1.val then
    s.writeMem r.dk (keyGradientAddress p l t k.1.val) (MaskedMath.dk p l initial r t k.1) else s) u

def scatterV (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (t : Nat) (u : BlockState) : BlockState :=
  (TileShape.allIndices [p.BV]).foldl (fun s j => if activeValue p l j.1.val then
    s.writeMem r.dv (outputAddress p l t j.1.val) (MaskedMath.dv p l initial r t j.1) else s) u

def retreatKey (p : Layout) (l : Launch) (r : RegionName) (t : Nat) : Tile .ptr [p.BK] :=
  pointers p.BK r (fun k => keyAddress p l t k.val-p.DK)
def retreatValue (p : Layout) (l : Launch) (r : RegionName) (t : Nat) : Tile .ptr [p.BV] :=
  pointers p.BV r (fun j => valueAddress p l t j.val-p.DV)
def retreatGradient (p : Layout) (l : Launch) (r : RegionName) (t : Nat) : Tile .ptr [p.BK] :=
  pointers p.BK r (fun k => keyGradientAddress p l t k.val-p.DK)
def retreatOutput (p : Layout) (l : Launch) (r : RegionName) (t : Nat) : Tile .ptr [p.BV] :=
  pointers p.BV r (fun j => outputAddress p l t j.val-p.DV)

theorem retreat_key_live (p : Layout) (l : Launch) (r : RegionName) (t : Nat) (ht : 0 < t) :
    retreatKey p l r t = keyPointers p l r (t-1) := by
  apply Tile.ext
  rintro ⟨k,⟨⟩⟩
  change (r,keyAddress p l t k.val-p.DK) = (r,keyAddress p l (t-1) k.val)
  apply congrArg (fun off : Nat => (r,off))
  exact BackwardRemainingMath.pointer_live _ p.DK t ht

theorem retreat_value_live (p : Layout) (l : Launch) (r : RegionName) (t : Nat) (ht : 0 < t) :
    retreatValue p l r t = valuePointers p l r (t-1) := by
  apply Tile.ext
  rintro ⟨j,⟨⟩⟩
  change (r,valueAddress p l t j.val-p.DV) = (r,valueAddress p l (t-1) j.val)
  apply congrArg (fun off : Nat => (r,off))
  exact BackwardRemainingMath.pointer_live _ p.DV t ht

theorem retreat_gradient_live (p : Layout) (l : Launch) (r : RegionName) (t : Nat) (ht : 0 < t) :
    retreatGradient p l r t = keyGradientPointers p l r (t-1) := by
  apply Tile.ext
  rintro ⟨k,⟨⟩⟩
  change (r,keyGradientAddress p l t k.val-p.DK) = (r,keyGradientAddress p l (t-1) k.val)
  apply congrArg (fun off : Nat => (r,off))
  exact BackwardRemainingMath.pointer_live _ p.DK t ht

theorem retreat_output_live (p : Layout) (l : Launch) (r : RegionName) (t : Nat) (ht : 0 < t) :
    retreatOutput p l r t = outputPointers p l r (t-1) := by
  apply Tile.ext
  rintro ⟨j,⟨⟩⟩
  change (r,outputAddress p l t j.val-p.DV) = (r,outputAddress p l (t-1) j.val)
  apply congrArg (fun off : Nat => (r,off))
  exact BackwardRemainingMath.pointer_live _ p.DV t ht

def result (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (t : Nat) (u : BlockState) : BlockState :=
  let u1 := scatterK p l initial r t (beforeStores p l initial r t u)
  let u2 := scatterV p l initial r t u1
  let u3 := u2.setReg "p_do" .ptr [p.BV] (retreatValue p l r.do_ t)
  let u4 := u3.setReg "p_q" .ptr [p.BK] (retreatKey p l r.q t)
  let u5 := u4.setReg "p_k" .ptr [p.BK] (retreatKey p l r.k t)
  let u6 := u5.setReg "p_v" .ptr [p.BV] (retreatValue p l r.v t)
  let u7 := u6.setReg "p_dk" .ptr [p.BK] (retreatGradient p l r.dk t)
  u7.setReg "p_dv" .ptr [p.BV] (retreatOutput p l r.dv t)

theorem ready_set_loop (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (remaining i : Nat) (u : BlockState) (hu : Ready p l initial r remaining u) :
    Ready p l initial r remaining (u.setReg "_t" .nat [] (Tile.scalar i)) := by
  rcases hu with ⟨hk,hv,hh,hb,hq,hkp,hvp,hd,hdk,hdv,hmk,hmv,hc⟩
  constructor <;> simp_all

theorem ready_result (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (t : Nat) (u : BlockState) (hu : Ready p l initial r (t+1) u) :
    Ready p l initial r t (result p l initial r t u) := by
  rcases hu with ⟨hk,hv,hh,hb,hq,hkp,hvp,hd,hdk,hdv,hmk,hmv,hc⟩
  constructor <;> simp [result,beforeStores,scatterK,scatterV,hk,hv,hh,hb,hmk,hmv]
  all_goals intro ht
  all_goals simp [retreat_key_live p l r.q t ht,retreat_key_live p l r.k t ht,
    retreat_value_live p l r.v t ht,retreat_value_live p l r.do_ t ht,
    retreat_gradient_live p l r.dk t ht,retreat_output_live p l r.dv t ht]

#axiomsClean retreat_key_live
#axiomsClean retreat_value_live
#axiomsClean retreat_gradient_live
#axiomsClean retreat_output_live
#axiomsClean ready_set_loop
#axiomsClean ready_result
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardSecondState
