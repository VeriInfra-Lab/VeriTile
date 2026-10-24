import bench.tritonbench_g.fused_recurrent_retention.Proofs.MaskedMath
import bench.tritonbench_g.fused_recurrent_retention.Proofs.Primitives

/-! The forward loop's internal invariant uses the mathematical state derived
from the initial input snapshot. It will be established by the source seed
prologue, not assumed in the public contract. The explicit intermediate states
below retain all actual register writes around the output scatter. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardState
open VeriTile.Triton Primitives MaskedMath
noncomputable section

def keyPointers (p : Layout) (l : Launch) (r : RegionName) (t : Nat) : Tile .ptr [p.BK] :=
  pointers p.BK r (fun j => keyAddress p l t j.val)
def valuePointers (p : Layout) (l : Launch) (r : RegionName) (t : Nat) : Tile .ptr [p.BV] :=
  pointers p.BV r (fun j => valueAddress p l t j.val)
def outputPointers (p : Layout) (l : Launch) (r : RegionName) (t : Nat) : Tile .ptr [p.BV] :=
  pointers p.BV r (fun j => outputAddress p l t j.val)

def carryTile (p : Layout) (l : Launch) (initial : BlockState) (r : ForwardRegions)
    (hasSeed : Bool) (t : Nat) : Tile .real [p.BV,p.BK] :=
  matrix p.BV p.BK (fun j k => stateValue p l initial r.k r.v r.initial hasSeed t k j)

structure Ready (p : Layout) (l : Launch) (initial : BlockState) (r : ForwardRegions)
    (hasSeed : Bool) (t : Nat) (u : BlockState) : Prop where
  keyId : u.regs .nat [] "i_k" = some (Tile.scalar l.key)
  valueId : u.regs .nat [] "i_v" = some (Tile.scalar l.value)
  headId : u.regs .nat [] "i_bh" = some (Tile.scalar l.head)
  decay : u.regs .real [] "b_b" = some (Tile.scalar (some (MaskedMath.decay p l)))
  q : u.regs .ptr [p.BK] "p_q" = some (keyPointers p l r.q t)
  k : u.regs .ptr [p.BK] "p_k" = some (keyPointers p l r.k t)
  v : u.regs .ptr [p.BV] "p_v" = some (valuePointers p l r.v t)
  o : u.regs .ptr [p.BV] "p_o" = some (outputPointers p l r.o t)
  maskK : u.regs .bool [p.BK] "mask_bk" =
    some (maskTile p.BK (fun j => activeKey p l j.val))
  maskV : u.regs .bool [p.BV] "mask_bv" =
    some (maskTile p.BV (fun j => activeValue p l j.val))
  maskKV : u.regs .bool [p.BV,p.BK] "mask_kv" =
    some (Tile.mat (fun j k => decide (activeKey p l k.val ∧ activeValue p l j.val)))
  carry : u.regs .real [p.BV,p.BK] "h" = some (carryTile p l initial r hasSeed t)

def beforeStore (p : Layout) (l : Launch) (initial : BlockState) (r : ForwardRegions)
    (hasSeed : Bool) (t : Nat) (u : BlockState) : BlockState :=
  let u1 := u.setReg "_k" .real [p.BK] (vector p.BK (keyValue p l initial r.k t))
  let u2 := u1.setReg "_v" .real [p.BV] (vector p.BV (valueValue p l initial r.v t))
  let u3 := u2.setReg "_q" .real [p.BK]
    (vector p.BK (fun k => keyValue p l initial r.q t k * p.scale))
  let u4 := u3.setReg "h" .real [p.BV,p.BK] (carryTile p l initial r hasSeed (t + 1))
  let u5 := u4.setReg "_o" .real [p.BV,p.BK] (matrix p.BV p.BK (fun j k =>
    stateValue p l initial r.k r.v r.initial hasSeed (t + 1) k j *
      (keyValue p l initial r.q t k * p.scale)))
  u5.setReg "_o" .real [p.BV] (vector p.BV (output p l initial r hasSeed t))

def scatter (p : Layout) (l : Launch) (initial : BlockState) (r : ForwardRegions)
    (hasSeed : Bool) (t : Nat) (u : BlockState) : BlockState :=
  (TileShape.allIndices [p.BV]).foldl (fun s j =>
    if activeValue p l j.1.val then
      s.writeMem r.o (outputAddress p l t j.1.val) (output p l initial r hasSeed t j.1)
    else s) u

def result (p : Layout) (l : Launch) (initial : BlockState) (r : ForwardRegions)
    (hasSeed : Bool) (t : Nat) (u : BlockState) : BlockState :=
  let u1 := scatter p l initial r hasSeed t (beforeStore p l initial r hasSeed t u)
  let u2 := u1.setReg "p_q" .ptr [p.BK] (keyPointers p l r.q (t + 1))
  let u3 := u2.setReg "p_k" .ptr [p.BK] (keyPointers p l r.k (t + 1))
  let u4 := u3.setReg "p_o" .ptr [p.BV] (outputPointers p l r.o (t + 1))
  u4.setReg "p_v" .ptr [p.BV] (valuePointers p l r.v (t + 1))

theorem ready_set_loop (p : Layout) (l : Launch) (initial : BlockState) (r : ForwardRegions)
    (hasSeed : Bool) (t i : Nat) (u : BlockState) (hu : Ready p l initial r hasSeed t u) :
    Ready p l initial r hasSeed t (u.setReg "_t" .nat [] (Tile.scalar i)) := by
  rcases hu with ⟨hk,hv,hh,hb,hq,hkptr,hvptr,hoptr,hmk,hmv,hmkv,hc⟩
  constructor <;> simp_all

theorem ready_result (p : Layout) (l : Launch) (initial : BlockState) (r : ForwardRegions)
    (hasSeed : Bool) (t : Nat) (u : BlockState) (hu : Ready p l initial r hasSeed t u) :
    Ready p l initial r hasSeed (t + 1) (result p l initial r hasSeed t u) := by
  rcases hu with ⟨hk,hv,hh,hb,hq,hkptr,hvptr,hoptr,hmk,hmv,hmkv,hc⟩
  constructor <;> simp [result, beforeStore, scatter, hk,hv,hh,hb,hq,hkptr,hvptr,hoptr,hmk,hmv,hmkv,hc]

#axiomsClean ready_set_loop
#axiomsClean ready_result
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardState
