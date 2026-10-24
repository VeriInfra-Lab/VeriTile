import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardHeaderSource
import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardFirstState
import bench.tritonbench_g.fused_recurrent_retention.Proofs.HeaderPrimitives

/-! Actual backward header/seed evaluation in the source's [BK,BV] order. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardHeaderPrimitives
open VeriTile.Triton Primitives MaskedMath BackwardFirstState BackwardHeaderSource
open ForwardHeaderSource (mulRef)
noncomputable section
set_option maxHeartbeats 900000
set_option linter.unusedSimpArgs false

def statePointers (p : Layout) (l : Launch) (r : RegionName) : Tile .ptr [p.BK,p.BV] :=
  Tile.mat fun k j => ⟨r,stateAddress p l k.val j.val⟩

def stateMask (p : Layout) (l : Launch) : Tile .bool [p.BK,p.BV] :=
  Tile.mat fun k j => decide (activeKey p l k.val ∧ activeValue p l j.val)

theorem gradient_pointer (p : Layout) (l : Launch) (r : RegionName) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hk : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hv : u.regs .nat [] "i_v" = some (Tile.scalar l.value)) :
    evalOp (gradientPointerOp p r) u = some (keyGradientPointers p l r 0) := by
  simp only [gradientPointerOp,mulRef,evalOp.eq_def,hh,hk,hv,Option.bind,bind,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨k,⟨⟩⟩
  simp [keyGradientPointers,keyGradientAddress,pointers,Tile.vec,Tile.ptrAdd,tile_elementwise,
    Broadcast.leftIndex,Broadcast.rightIndex]

theorem state_mask (p : Layout) (l : Launch) (u : BlockState)
    (hk : u.regs .bool [p.BK] "mask_bk" = some (maskTile p.BK (fun k => activeKey p l k.val)))
    (hv : u.regs .bool [p.BV] "mask_bv" = some (maskTile p.BV (fun j => activeValue p l j.val))) :
    evalOp (maskKV p) u = some (stateMask p l) := by
  simp only [maskKV,evalOp.eq_def,hk,hv,Option.bind,bind,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨k,j,⟨⟩⟩
  simp [stateMask,maskTile,Tile.mat,Tile.vec,Tile.expandDim,Tile.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex]

theorem state_pointer (p : Layout) (l : Launch) (r : RegionName) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hk : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hv : u.regs .nat [] "i_v" = some (Tile.scalar l.value)) :
    evalOp (statePointerOp p r) u = some (statePointers p l r) := by
  simp only [statePointerOp,mulRef,evalOp.eq_def,hh,hk,hv,Option.bind,bind,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨k,j,⟨⟩⟩
  simp [statePointers,stateAddress,Tile.mat,Tile.ptrAdd,tile_elementwise,
    Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex]

theorem seed_load (p : Layout) (l : Launch) (r : RegionName) (initial u : BlockState)
    (hp : u.regs .ptr [p.BK,p.BV] "p_init_s" = some (statePointers p l r))
    (hm : u.regs .bool [p.BK,p.BV] "mask_kv" = some (stateMask p l))
    (hmem : ∀ off, u.readMem r off = initial.readMem r off) :
    evalOp (seedLoad p) u = some (matrix p.BK p.BV (fun k j => seed p l initial r Bool.true k j)) := by
  simp only [seedLoad,evalOp.eq_def,hp,hm,Option.bind,bind,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨k,j,⟨⟩⟩
  by_cases hk : activeKey p l k.val <;> by_cases hv : activeValue p l j.val <;>
    simp [statePointers,stateMask,matrix,Tile.mat,seed,hk,hv,BlockState.readMemValue_real,hmem]

theorem seed_add (p : Layout) (l : Launch) (r : BackwardRegions) (initial u : BlockState)
    (hp : u.regs .ptr [p.BK,p.BV] "p_init_s" = some (statePointers p l r.initial))
    (hm : u.regs .bool [p.BK,p.BV] "mask_kv" = some (stateMask p l))
    (hh : u.regs .real [p.BK,p.BV] "h" = some (matrix p.BK p.BV (fun _ _ => 0)))
    (hmem : ∀ off, u.readMem r.initial off = initial.readMem r.initial off) :
    evalOp (seedAdd p) u = some (carryTile p l initial r Bool.true 0) := by
  rw [seedAdd,evalOp_add,evalOp_ref,hh,seed_load p l r.initial initial u hp hm hmem]
  simp only [Option.bind,bind,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨k,j,⟨⟩⟩
  simp [carryTile,state_zero,matrix,Tile.mat,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex]

theorem unseeded_zero (p : Layout) (l : Launch) (r : BackwardRegions) (initial : BlockState) :
    carryTile p l initial r Bool.false 0 = matrix p.BK p.BV (fun _ _ => 0) := by
  simp [carryTile,state_zero,seed]

#axiomsClean gradient_pointer
#axiomsClean state_mask
#axiomsClean state_pointer
#axiomsClean seed_load
#axiomsClean seed_add
#axiomsClean unseeded_zero
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardHeaderPrimitives
