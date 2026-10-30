import bench.repairs.tritonbench_g.chunk_retention.Proofs.PureAccessSafety
import bench.repairs.tritonbench_g.chunk_retention.Proofs.ForwardHComplete
import bench.repairs.tritonbench_g.chunk_retention.Proofs.AccessSafetyLoops

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardHSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 800000
set_option linter.unusedSimpArgs false
attribute [local simp 2000] AccessSafety.two_offset_pointer

theorem finish_safe (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (seed final : Bool) (initial current : State α) (n : Nat) (carry : Index c → α)
    (ctx : Context M c initial current n) (ha : HasCarry c carry current)
    (bounds : RegionName → Nat)
    (ht : final=Bool.true → ∀ i,active c initial i → stateAddress c initial i<bounds "ht") :
    Safety.traceSafeList (engine (withDot M D)) bounds (parts c seed final).after current := by
  simp only [HasCarry] at ha
  generalize hbackend : engine (withDot M D)=E
  change Safety.traceSafeList E bounds [_] current
  cases final
  all_goals
    simp [AccessSafety.singleton_safe,AccessSafety.pair_safe,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,numericEq,natLe,numericLe,numericLt,natLt,bop,
      ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
      Broadcast.leftIndex,Broadcast.rightIndex,ctx.ik,ctx.iv,ctx.bh,ha,FP.Structural.store]
  intro e j he hj
  exact ht rfl (e,j,PUnit.unit) ⟨he,hj⟩


#axiomsClean finish_safe

end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardHSource
