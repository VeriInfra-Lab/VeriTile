import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardMiddleSource
import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardSecondState

/-! Numerical-free execution of every source pointer rebase, plus the actual
barrier projection and zero initialization. No positive-time premise is used. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardMiddlePrimitives
open VeriTile.Triton Primitives MaskedMath BackwardMiddleSource
open ForwardHeaderSource (mulRef)
open ForwardState (keyPointers valuePointers outputPointers)
open BackwardFirstState (keyGradientPointers)
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

theorem pointer (p : Layout) (n headStride stride head idx : Nat)
    (name : RegName) (r : RegionName) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar head))
    (hi : u.regs .nat [] name = some (Tile.scalar idx)) :
    evalOp (pointerOp p n headStride stride name r) u =
      some (pointers n r (fun j => head*headStride+idx*n+j.val+(p.T-1)*stride)) := by
  simp only [pointerOp,lastOffset,mulRef,evalOp.eq_def,hh,hi,Option.bind,bind,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨j,⟨⟩⟩
  simp [pointers,Tile.vec,Tile.ptrAdd,tile_elementwise,
    Broadcast.leftIndex,Broadcast.rightIndex]

theorem gradient_pointer (p : Layout) (l : Launch) (r : RegionName) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hk : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hv : u.regs .nat [] "i_v" = some (Tile.scalar l.value)) :
    evalOp (gradientPointerOp p r) u = some (keyGradientPointers p l r (p.T-1)) := by
  simp only [gradientPointerOp,lastOffset,mulRef,evalOp.eq_def,hh,hk,hv,Option.bind,bind,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨k,⟨⟩⟩
  simp [keyGradientPointers,keyGradientAddress,pointers,Tile.vec,Tile.ptrAdd,tile_elementwise,
    Broadcast.leftIndex,Broadcast.rightIndex]

theorem output_pointer (p : Layout) (l : Launch) (r : RegionName) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hk : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hv : u.regs .nat [] "i_v" = some (Tile.scalar l.value)) :
    evalOp (outputPointerOp p r) u = some (outputPointers p l r (p.T-1)) := by
  simp only [outputPointerOp,lastOffset,mulRef,evalOp.eq_def,hh,hk,hv,Option.bind,bind,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨j,⟨⟩⟩
  simp [outputPointers,outputAddress,pointers,Tile.vec,Tile.ptrAdd,tile_elementwise,
    Broadcast.leftIndex,Broadcast.rightIndex]

theorem barrier (u : BlockState) : stepStmt (.ifThen (.constBool Bool.false) []) u = some u := by
  have hc : evalOp (.constBool Bool.false) u = some (Tile.scalar Bool.false) := by
    simp only [evalOp.eq_def]
  simp only [stepStmt,hc,Tile.scalar_data,Option.bind,bind,Bool.false_eq_true,↓reduceIte]

theorem zero_carry (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (u : BlockState) :
    evalOp (.full [p.BK,p.BV] (.const 0)) u =
      some (BackwardSecondState.carryTile p l initial r p.T) := by
  simp only [evalOp_full,evalOp_const,Option.bind,bind,Option.pure_def]
  have he : BackwardSecondState.carryTile p l initial r p.T = matrix p.BK p.BV (fun _ _ => 0) := by
    unfold BackwardSecondState.carryTile
    apply congrArg (matrix p.BK p.BV)
    funext k j
    exact BackwardRemainingMath.at_end p l initial r k j
  rw [he]
  rfl

#axiomsClean pointer
#axiomsClean gradient_pointer
#axiomsClean output_pointer
#axiomsClean barrier
#axiomsClean zero_carry
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardMiddlePrimitives
