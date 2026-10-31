import bench.tritonbench_g.chunk_retention.Sources.ForwardHComplete
import bench.tritonbench_g.chunk_retention.Proofs.AccessSafetyLoops
import bench.tritonbench_g.chunk_retention.Proofs.PureAccessSafety

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardHSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 1500000
set_option linter.unusedSimpArgs false
attribute [local simp 2000] AccessSafety.two_offset_pointer

/-- The optional initial-state read follows the source's block-pointer boundary
checks; every preceding arithmetic statement is memory-free. -/
theorem entry_safe (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (seed final : Bool) (s : State α) (bounds : RegionName → Nat)
    (h0 : seed=Bool.true → ∀ i,active c s i → stateAddress c s i<bounds "h0") :
    Safety.traceSafeList (engine (withDot M D)) bounds (parts c seed final).before s := by
  cases seed
  all_goals
    simp [parts,kernel,Kernels.forwardH,ComputeKernel.surfaceBody,Source.dynamicRange?,
      Source.prefixBefore,Option.map,Option.getD,AccessSafety.singleton_safe,AccessSafety.pair_safe,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,KernelModel.signed_head_mod,Int.toNat_mul,Int.toNat_add,Int.toNat_natCast,natLe,numericLe,numericLt,natLt,bop,
      ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  intro e j he hj
  exact h0 rfl (e,j,PUnit.unit) ⟨he,hj⟩

#axiomsClean entry_safe

end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardHSource
