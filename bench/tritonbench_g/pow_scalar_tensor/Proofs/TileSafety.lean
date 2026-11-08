import bench.tritonbench_g.pow_scalar_tensor.Proofs.Source
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked.SourceSafety
open VeriTile Triton FP.Structural Safety
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Bounds apply to the actual signed block addresses. Inactive pointers may
be negative; every active access is checked against its backing allocation. -/
theorem tile_safe (M : Algebra α) (c : Parameters) (candidate : Bool) (s : State α) (T : Nat)
    (ht : s.regs .nat [] "tile_id"=some (fun _ => T)) (bounds : RegionName → Nat)
    (hi : ∀ j : Fin c.B,T*c.B+j.val<c.n →
      (c.input.base+((T*c.B+j.val:Nat):Int)*c.input.stride).toNat<bounds c.input.region)
    (ho : ∀ j : Fin c.B,T*c.B+j.val<c.n →
      (c.output.base+((T*c.B+j.val:Nat):Int)*c.output.stride).toNat<bounds c.output.region) :
    traceSafeList M bounds (perTile c candidate) s := by
  cases candidate <;> cases branch : c.oneTile
  all_goals
    simp only [perTile,PastedTile.body,branch,Bool.false_eq_true,if_false,if_true,
      PastedTile.oneTile,PastedTile.grid,PastedTile.iterations,oneTileKernel,oneTileCommuted,
      gridStrideKernel,gridStrideCommuted,originalKernel,commutedBodyPastedKernel,ComputeKernel.surfaceBody]
    simp [traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
      safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      step,stepAlg,evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,numericLe,numericEq,natLt,
      FP.Structural.bop,store,ht,ComputeDType.eraseDType,Int.not_ofNat_neg,
      Broadcast.leftIndex,Broadcast.rightIndex,
      ← Int.natCast_add,← Int.natCast_mul,-Nat.cast_add,-Nat.cast_mul,
      FloatDType.comparableDType,Region.cast,ofFloat,toFloat]
    constructor
    · intro i active
      simpa only [Nat.mul_comm,Int.mul_comm] using hi i (by simpa only [Nat.mul_comm] using active)
    · intro i active
      simpa only [Nat.mul_comm,Int.mul_comm] using ho i (by simpa only [Nat.mul_comm] using active)
#axiomsClean tile_safe
end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked.SourceSafety
