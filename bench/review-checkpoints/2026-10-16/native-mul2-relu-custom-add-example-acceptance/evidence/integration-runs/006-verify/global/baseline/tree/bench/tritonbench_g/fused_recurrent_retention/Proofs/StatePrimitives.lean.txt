import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardState

/-! Actual seed load and zero-plus-seed update. Masked key/value tails load
the literal zero from the source, while active lanes read the initial snapshot. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.StatePrimitives
open VeriTile.Triton Primitives MaskedMath ForwardState
noncomputable section
set_option maxHeartbeats 700000
set_option linter.unusedSimpArgs false

def statePointers (p : Layout) (l : Launch) (r : RegionName) : Tile .ptr [p.BV,p.BK] :=
  Tile.mat fun j k => ⟨r,stateAddress p l k.val j.val⟩

def stateMask (p : Layout) (l : Launch) : Tile .bool [p.BV,p.BK] :=
  Tile.mat fun j k => decide (activeKey p l k.val ∧ activeValue p l j.val)

def seedLoad (p : Layout) : Op .real [p.BV,p.BK] :=
  .load .real (.ptr (.ref .ptr [p.BV,p.BK] "p_init_s"))
    (.maskOther (.ref .bool [p.BV,p.BK] "mask_kv")
      (.broadcast (.const 0) [p.BV,p.BK]))

def seedAdd (p : Layout) : Op .real [p.BV,p.BK] :=
  .add .real (.consSame (.consSame .nil)) (.ref .real [p.BV,p.BK] "h") (seedLoad p)

theorem seed_load (p : Layout) (l : Launch) (r : RegionName) (initial u : BlockState)
    (hp : u.regs .ptr [p.BV,p.BK] "p_init_s" = some (statePointers p l r))
    (hm : u.regs .bool [p.BV,p.BK] "mask_kv" = some (stateMask p l))
    (hmem : ∀ off, u.readMem r off = initial.readMem r off) :
    evalOp (seedLoad p) u = some (matrix p.BV p.BK
      (fun j k => seed p l initial r Bool.true k j)) := by
  simp only [seedLoad,evalOp.eq_def,hp,hm,Option.bind,bind,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨j,k,⟨⟩⟩
  by_cases hk : activeKey p l k.val <;> by_cases hv : activeValue p l j.val <;>
    simp [statePointers,stateMask,matrix,Tile.mat,seed,hk,hv,
      BlockState.readMemValue_real,hmem]

theorem seed_add (p : Layout) (l : Launch) (r : ForwardRegions)
    (initial u : BlockState)
    (hp : u.regs .ptr [p.BV,p.BK] "p_init_s" = some (statePointers p l r.initial))
    (hm : u.regs .bool [p.BV,p.BK] "mask_kv" = some (stateMask p l))
    (hh : u.regs .real [p.BV,p.BK] "h" = some (matrix p.BV p.BK (fun _ _ => 0)))
    (hmem : ∀ off, u.readMem r.initial off = initial.readMem r.initial off) :
    evalOp (seedAdd p) u = some (carryTile p l initial r Bool.true 0) := by
  rw [seedAdd,evalOp_add,evalOp_ref,hh,seed_load p l r.initial initial u hp hm hmem]
  simp only [Option.bind,bind,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨j,k,⟨⟩⟩
  simp [carryTile,state_zero,matrix,Tile.mat,tile_elementwise,
    Broadcast.leftIndex,Broadcast.rightIndex]

theorem unseeded_zero (p : Layout) (l : Launch) (r : ForwardRegions) (initial : BlockState) :
    carryTile p l initial r Bool.false 0 = matrix p.BV p.BK (fun _ _ => 0) := by
  simp [carryTile,state_zero,seed]

#axiomsClean seed_load
#axiomsClean seed_add
#axiomsClean unseeded_zero
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.StatePrimitives
