import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Tile
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.L2NormBwd.Tile
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Bounds of the actual masked X/DY loads and DX store in the original
backward source. Runtime epsilon affects arithmetic, never these addresses. -/
theorem source_traceSafe {A : Type} [Inhabited A] (M : Algebra A)
    (layout : Strides) (N B row : Nat) (eps : A) (bounds : RegionName → Nat) (s : State A)
    (input : ∀ j : Fin B, j.val<N → row*layout.xRow+j.val*layout.xCol < bounds "X")
    (gradient : ∀ j : Fin B, j.val<N → row*layout.dyRow+j.val*layout.dyCol < bounds "DY")
    (output : ∀ j : Fin B, j.val<N → row*layout.dxRow+j.val*layout.dxCol < bounds "DX") :
    FP.Structural.TraceSafe M bounds (source layout N B) (rowState row eps s) := by
  have index_toNat (a b lane : Nat) :
      ((row : Int)*(a : Int)+(lane : Int)*(b : Int)).toNat=row*a+lane*b := by
    rw [←Int.natCast_mul,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]
  have index_lt (lane : Nat) : ((lane : Int)<(N : Int)) ↔ lane<N := by omega
  simp (maxSteps := 1600000) [FP.Structural.TraceSafe,source,kernel,
    ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,step,FP.Structural.evalExpr,evalComputeOp,
    evalOp_unfold,evalOffsets,evalDeltas,store,Option.bind,
    ComputeDType.eraseDType,numeric,numericLt,bop,natLt,rowState,
    ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,index_toNat,index_lt]
  exact ⟨input,gradient,output⟩

#axiomsClean source_traceSafe
end
end VeriTile.Bench.Repairs.L2NormBwd.Tile
