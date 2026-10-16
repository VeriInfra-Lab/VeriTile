import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardHeaderSource

/-! Successful evaluation of every header expression from its actual scalar
program-id registers. Pointer arithmetic remains source-ordered and unbounded
in this mathematical interpreter; a fixed-width native bridge is separate. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.HeaderPrimitives
open VeriTile.Triton Primitives MaskedMath ForwardState ForwardHeaderSource StatePrimitives
noncomputable section
set_option maxHeartbeats 900000
set_option linter.unusedSimpArgs false

theorem head_index (p : Layout) (l : Launch) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head)) :
    evalOp (.tritonMod .nat .nil (.ref .nat [] "i_bh") (.constNat p.H)) u =
      some (Tile.scalar (l.head % p.H)) := by
  simp only [Op.tritonMod,evalOp.eq_def,hh,Option.bind,bind,Option.pure_def]
  rfl

set_option maxRecDepth 2000 in
theorem decay (p : Layout) (l : Launch) (u : BlockState)
    (hh : u.regs .nat [] "i_h" = some (Tile.scalar (l.head % p.H))) :
    evalOp decayOp u = some (Tile.scalar (some (MaskedMath.decay p l))) := by
  let x : Op .real [] :=
    .sub .real .nil (.const (-5))
      (.mul .real .nil (.natToReal (.ref .nat [] "i_h")) (.const 1.0))
  have hn : evalOp (.natToReal (.ref .nat [] "i_h")) u =
      some (Tile.scalar (some (((l.head % p.H : Nat) : ℝ)))) := by
    rw [evalOp_natToReal,evalOp_ref,hh]
    rfl
  have hm : evalOp (.mul .real .nil (.natToReal (.ref .nat [] "i_h")) (.const 1.0)) u =
      some (Tile.scalar (some (((l.head % p.H : Nat) : ℝ)))) := by
    rw [evalOp_mul,hn,evalOp_const]
    simp only [Option.bind_some,Option.bind_eq_bind]
    apply congrArg some
    apply Tile.ext
    rintro ⟨⟩
    change (some ((((l.head % p.H : Nat) : ℝ)) * 1.0) : WithBot ℝ) =
      some (((l.head % p.H : Nat) : ℝ))
    norm_num
  have hz : evalOp (.const (-5)) u =
      some (Tile.scalar (some (-5 : ℝ))) := evalOp_const (-5) u
  have hx : evalOp x u = some (Tile.scalar (some (-5 - ((l.head % p.H : Nat) : ℝ)))) := by
    dsimp only [x]
    rw [evalOp_sub,hz,hm]
    rfl
  have he : evalOp (.exp2 x) u = some (Tile.scalar
      (some (Real.exp ((-5 - ((l.head % p.H : Nat) : ℝ)) * Real.log 2)))) := by
    rw [evalOp,hx]
    rfl
  change evalOp (.sub .real .nil (.const 1) (.exp2 x)) u = _
  rw [evalOp_sub,evalOp_const,he]
  rfl

theorem pointer (n stride head idx : Nat) (name : RegName) (r : RegionName) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar head))
    (hi : u.regs .nat [] name = some (Tile.scalar idx)) :
    evalOp (pointerOp n stride name r) u =
      some (pointers n r (fun j => head*stride+idx*n+j.val)) := by
  simp only [pointerOp,mulRef,evalOp.eq_def,hh,hi,Option.bind,bind,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨j,⟨⟩⟩
  simp [pointers,Tile.vec,Tile.ptrAdd,tile_elementwise,
    Broadcast.leftIndex,Broadcast.rightIndex]

theorem output_pointer (p : Layout) (l : Launch) (r : RegionName) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hk : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hv : u.regs .nat [] "i_v" = some (Tile.scalar l.value)) :
    evalOp (outputPointerOp p r) u = some (outputPointers p l r 0) := by
  simp only [outputPointerOp,mulRef,evalOp.eq_def,hh,hk,hv,Option.bind,bind,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨j,⟨⟩⟩
  simp [outputPointers,outputAddress,pointers,Tile.vec,Tile.ptrAdd,tile_elementwise,
    Broadcast.leftIndex,Broadcast.rightIndex]

theorem key_mask (p : Layout) (l : Launch) (swapped : Bool) (u : BlockState)
    (hk : u.regs .nat [] "i_k" = some (Tile.scalar l.key)) :
    evalOp (keyMask p swapped) u = some (maskTile p.BK (fun j => activeKey p l j.val)) := by
  cases swapped
  all_goals simp only [keyMask,mulRef,Bool.false_eq_true,↓reduceIte,evalOp.eq_def,
    hk,Option.bind,bind,Option.pure_def]
  all_goals apply congrArg some
  all_goals apply Tile.ext
  all_goals rintro ⟨j,⟨⟩⟩
  all_goals simp [maskTile,activeKey,Tile.vec,tile_elementwise,
    Broadcast.leftIndex,Broadcast.rightIndex]

theorem value_mask (p : Layout) (l : Launch) (u : BlockState)
    (hv : u.regs .nat [] "i_v" = some (Tile.scalar l.value)) :
    evalOp (valueMask p) u = some (maskTile p.BV (fun j => activeValue p l j.val)) := by
  simp only [valueMask,mulRef,evalOp.eq_def,hv,Option.bind,bind,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨j,⟨⟩⟩
  simp [maskTile,activeValue,Tile.vec,tile_elementwise,
    Broadcast.leftIndex,Broadcast.rightIndex]

theorem state_mask (p : Layout) (l : Launch) (u : BlockState)
    (hk : u.regs .bool [p.BK] "mask_bk" = some (maskTile p.BK (fun j => activeKey p l j.val)))
    (hv : u.regs .bool [p.BV] "mask_bv" = some (maskTile p.BV (fun j => activeValue p l j.val))) :
    evalOp (maskKV p) u = some (stateMask p l) := by
  simp only [maskKV,evalOp.eq_def,hk,hv,Option.bind,bind,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨j,k,⟨⟩⟩
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
  rintro ⟨j,k,⟨⟩⟩
  simp [statePointers,stateAddress,Tile.mat,Tile.ptrAdd,tile_elementwise,
    Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex]

#axiomsClean head_index
#axiomsClean decay
#axiomsClean pointer
#axiomsClean output_pointer
#axiomsClean key_mask
#axiomsClean value_mask
#axiomsClean state_mask
#axiomsClean state_pointer
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.HeaderPrimitives
