import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.SafetySteps
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked.SourceSafety
open VeriTile.Triton FP.Structural
open TritonBenchG.ReluStridedBuffer SignedView SourceSignedBranch
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Bounds are checked at actual signed-address loads and stores, after the
complete signed offset is converted to a cell address. Masked lanes impose no
backing requirement, including negative inactive pointers. -/
theorem tile_safe {α : Type} [Inhabited α] (M : Algebra α) (c : Parameters) (candidate : Bool) (s : State α) (T : Nat)
    (ht : s.regs .nat [] "tile_id"=some (fun _ => T)) (bounds : RegionName → Nat)
    (input : ∀ j : Fin c.B,T*c.B+j.val<c.n →
      (c.input.base+((T*c.B+j.val:Nat):Int)*c.input.stride).toNat<bounds c.input.region)
    (output : ∀ j : Fin c.B,T*c.B+j.val<c.n →
      (c.output.base+((T*c.B+j.val:Nat):Int)*c.output.stride).toNat<bounds c.output.region) :
    Safety.traceSafeList M bounds (perTile c candidate) s := by
  cases candidate
  all_goals simp [perTile,oneTileKernel,oneTileCommuted,originalBody,commutedBody,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,stepAlg,evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,numericLe,natLt,natLe,FP.Structural.bop,
    store,ht,Nat.mul_comm,Int.natCast_add,Int.natCast_mul,FloatDType.comparableDType,Region.cast,ofFloat,toFloat]
  all_goals simp only [←Int.natCast_mul,←Int.natCast_add,Int.natCast_nonneg,
    decide_true,Bool.true_and,true_and,true_implies,Nat.cast_lt]
  all_goals constructor
  all_goals intro i active
  all_goals first | exact input i active | exact output i active

#axiomsClean tile_safe
end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked.SourceSafety
