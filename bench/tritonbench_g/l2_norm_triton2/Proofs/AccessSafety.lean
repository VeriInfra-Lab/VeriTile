import bench.tritonbench_g.l2_norm_triton2.Proofs.Tile
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.L2NormTriton2.Tile
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- The original backward source accesses exactly the two input row windows
and the output row, retaining the actual runtime epsilon register. -/
theorem source_traceSafe {α : Type} [Inhabited α] (M : Algebra α)
    (N B row : Nat) (eps : α) (bounds : RegionName → Nat) (s : State α)
    (input : ∀ j : Fin B,j.val<N → row*N+j.val<bounds "X")
    (gradient : ∀ j : Fin B,j.val<N → row*N+j.val<bounds "DY")
    (output : ∀ j : Fin B,j.val<N → row*N+j.val<bounds "DX") :
    TraceSafe M bounds (backward N B) (rowState row eps s) := by
  have index_toNat (lane : Nat) :
      ((row : Int)*(N : Int)+(lane : Int)).toNat=row*N+lane := by
    rw [←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]
  have index_lt (lane : Nat) : ((lane : Int)<(N : Int)) ↔ lane<N := by omega
  simp (maxSteps := 1800000) [TraceSafe,backward,backwardTemplate,
    ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,step,FP.Structural.evalExpr,evalComputeOp,
    evalOp_unfold,evalOffsets,evalDeltas,store,Option.bind,
    ComputeDType.eraseDType,numeric,numericLt,bop,natLt,rowState,
    ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,index_toNat,index_lt]
  exact ⟨input,gradient,output⟩

/-- The existing index-product candidate has the same actual safe accesses. -/
theorem commuted_traceSafe {α : Type} [Inhabited α] (M : Algebra α)
    (N B row : Nat) (eps : α) (bounds : RegionName → Nat) (s : State α)
    (input : ∀ j : Fin B,j.val<N → row*N+j.val<bounds "X")
    (gradient : ∀ j : Fin B,j.val<N → row*N+j.val<bounds "DY")
    (output : ∀ j : Fin B,j.val<N → row*N+j.val<bounds "DX") :
    TraceSafe M bounds (backwardCommuted N B) (rowState row eps s) := by
  have index_toNat (lane : Nat) :
      ((N : Int)*(row : Int)+(lane : Int)).toNat=row*N+lane := by
    rw [Int.mul_comm,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]
  have index_product : ((N : Int)*(row : Int)).toNat = row*N := by
    rw [←Int.natCast_mul,Int.toNat_natCast,Nat.mul_comm]
  have index_lt (lane : Nat) : ((lane : Int)<(N : Int)) ↔ lane<N := by omega
  simp (maxSteps := 1800000) [TraceSafe,backwardCommuted,commutedBackwardTemplate,
    ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,step,FP.Structural.evalExpr,evalComputeOp,
    evalOp_unfold,evalOffsets,evalDeltas,store,Option.bind,
    ComputeDType.eraseDType,numeric,numericLt,bop,natLt,rowState,
    ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,index_toNat,index_product,index_lt]
  exact ⟨input,gradient,output⟩

#axiomsClean source_traceSafe
#axiomsClean commuted_traceSafe
end
end VeriTile.Bench.Repairs.L2NormTriton2.Tile
