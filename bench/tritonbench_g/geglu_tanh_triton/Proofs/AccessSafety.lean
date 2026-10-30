import bench.tritonbench_g.geglu_tanh_triton.Proofs.Tile
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.GegluTanhTriton.AccessSafety
open VeriTile Triton FP.Structural Safety
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

private theorem index_toNat (row block lane : Nat) :
    ((row : Int)*(block : Int)+(lane : Int)).toNat=row*block+lane := by
  rw [←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]

private theorem index_lt (row block lane count : Nat) :
    ((row : Int)*(block : Int)+(lane : Int)<(count : Int)) ↔ row*block+lane<count := by
  rw [←Int.natCast_mul,←Int.natCast_add]
  omega

/-- Every actual masked read and write of either complete source variant.
The backward source loads A/B/DC before its in-place A/B stores. -/
theorem source_safe {α : Type} [Inhabited α] (M : Algebra α)
    (candidate backward : Bool) (count block : Nat) (bounds : RegionName → Nat) (s : State α)
    (inputs : ∀ i,i<count → i<bounds "A" ∧ i<bounds "B" ∧ (backward=Bool.true → i<bounds "DC"))
    (outputs : ∀ ch,enabled backward ch → ∀ i,i<count → i<bounds (outputRegion ch)) :
    TraceSafe M bounds (selectedSource backward candidate block count) s := by
  cases candidate <;> cases backward
  all_goals
    simp (config := {maxSteps:=500000}) [TraceSafe,selectedSource,source,indexCommutedSource,
      forwardKernel,backwardKernel,forwardIndexCommuted,backwardIndexCommuted,
      ComputeKernel.surfaceBody,
      traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
      safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,Option.bind,
      ComputeDType.eraseDType,numeric,numericLt,bop,natLt,ofFloat,toFloat,TileShape.axisDim,
      Region.cast,State.setReg,index_toNat,index_lt,Int.mul_comm]
  all_goals
    constructor
    · intro i active
      first | exact (inputs _ active).1 | exact (inputs _ active).2.2 rfl
    · constructor
      · intro i active
        first | exact (inputs _ active).2.1 | exact (inputs _ active).1
      · first
        | exact fun i active=>outputs .c rfl _ active
        | refine ⟨fun i active=>(inputs _ active).2.1,
            fun i active=>outputs .da (by decide) _ active,
            fun i active=>outputs .db (by decide) _ active⟩

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.GegluTanhTriton.AccessSafety
