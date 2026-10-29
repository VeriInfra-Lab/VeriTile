import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardPrimitives
import bench.tritonbench_g.fused_recurrent_retention.Proofs.MaskedMath
import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardState

/-! The first backward loop reconstructs its complete transposed carry from
initial inputs. These internal register invariants will be established by the
actual source seed prologue; none is a public input/carry assumption. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardFirstState
open VeriTile.Triton Primitives MaskedMath ForwardState
noncomputable section
set_option linter.unusedSimpArgs false

def keyGradientPointers (p : Layout) (l : Launch) (r : RegionName) (t : Nat) : Tile .ptr [p.BK] :=
  pointers p.BK r (fun k => keyGradientAddress p l t k.val)

def carryTile (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (hasSeed : Bool) (t : Nat) : Tile .real [p.BK,p.BV] :=
  matrix p.BK p.BV (fun k j => stateValue p l initial r.k r.v r.initial hasSeed t k j)

structure Ready (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (hasSeed : Bool) (t : Nat) (u : BlockState) : Prop where
  keyId : u.regs .nat [] "i_k" = some (Tile.scalar l.key)
  valueId : u.regs .nat [] "i_v" = some (Tile.scalar l.value)
  headId : u.regs .nat [] "i_bh" = some (Tile.scalar l.head)
  decay : u.regs .real [] "b_b" = some (Tile.scalar (some (MaskedMath.decay p l)))
  k : u.regs .ptr [p.BK] "p_k" = some (keyPointers p l r.k t)
  v : u.regs .ptr [p.BV] "p_v" = some (valuePointers p l r.v t)
  dout : u.regs .ptr [p.BV] "p_do" = some (valuePointers p l r.do_ t)
  dq : u.regs .ptr [p.BK] "p_dq" = some (keyGradientPointers p l r.dq t)
  maskK : u.regs .bool [p.BK] "mask_bk" = some (maskTile p.BK (fun k => activeKey p l k.val))
  maskV : u.regs .bool [p.BV] "mask_bv" = some (maskTile p.BV (fun j => activeValue p l j.val))
  carry : u.regs .real [p.BK,p.BV] "h" = some (carryTile p l initial r hasSeed t)

def beforeStore (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (hasSeed : Bool) (t : Nat) (u : BlockState) : BlockState :=
  let u1 := u.setReg "_k" .real [p.BK] (vector p.BK (keyValue p l initial r.k t))
  let u2 := u1.setReg "_v" .real [p.BV] (vector p.BV (valueValue p l initial r.v t))
  let u3 := u2.setReg "_do" .real [p.BV] (vector p.BV (valueValue p l initial r.do_ t))
  let u4 := u3.setReg "h" .real [p.BK,p.BV] (carryTile p l initial r hasSeed (t+1))
  let u5 := u4.setReg "_d_q" .real [p.BK,p.BV] (matrix p.BK p.BV (fun k j =>
    stateValue p l initial r.k r.v r.initial hasSeed (t+1) k j * valueValue p l initial r.do_ t j))
  u5.setReg "d_q" .real [p.BK] (vector p.BK (MaskedMath.dq p l initial r hasSeed t))

def scatter (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (hasSeed : Bool) (t : Nat) (u : BlockState) : BlockState :=
  (TileShape.allIndices [p.BK]).foldl (fun s k =>
    if activeKey p l k.1.val then
      s.writeMem r.dq (keyGradientAddress p l t k.1.val) (MaskedMath.dq p l initial r hasSeed t k.1)
    else s) u

def result (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (hasSeed : Bool) (t : Nat) (u : BlockState) : BlockState :=
  let u1 := scatter p l initial r hasSeed t (beforeStore p l initial r hasSeed t u)
  let u2 := u1.setReg "p_k" .ptr [p.BK] (keyPointers p l r.k (t+1))
  let u3 := u2.setReg "p_do" .ptr [p.BV] (valuePointers p l r.do_ (t+1))
  let u4 := u3.setReg "p_v" .ptr [p.BV] (valuePointers p l r.v (t+1))
  u4.setReg "p_dq" .ptr [p.BK] (keyGradientPointers p l r.dq (t+1))

theorem ready_set_loop (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (hasSeed : Bool) (t i : Nat) (u : BlockState) (hu : Ready p l initial r hasSeed t u) :
    Ready p l initial r hasSeed t (u.setReg "i" .nat [] (Tile.scalar i)) := by
  rcases hu with ⟨hk,hv,hh,hb,hkptr,hvptr,hdptr,hqptr,hmk,hmv,hc⟩
  constructor <;> simp_all

theorem ready_result (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (hasSeed : Bool) (t : Nat) (u : BlockState) (hu : Ready p l initial r hasSeed t u) :
    Ready p l initial r hasSeed (t+1) (result p l initial r hasSeed t u) := by
  rcases hu with ⟨hk,hv,hh,hb,hkptr,hvptr,hdptr,hqptr,hmk,hmv,hc⟩
  constructor <;> simp [result,beforeStore,scatter,hk,hv,hh,hb,hkptr,hvptr,hdptr,hqptr,hmk,hmv,hc]

#axiomsClean ready_set_loop
#axiomsClean ready_result
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardFirstState
