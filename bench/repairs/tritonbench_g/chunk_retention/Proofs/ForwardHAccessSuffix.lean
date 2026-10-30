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

theorem suffix_safe (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (seed final : Bool) (initial current : State α) (n : Nat) (carry : Index c → α)
    (hn : n<c.chunks) (ctx : Prepared M c initial current n carry)
    (bounds : RegionName → Nat)
    (hk : ∀ e t,e<c.K → t<c.T →
      initial.pids 2*(c.T*c.K)+e+t*c.K<bounds "k")
    (hv : ∀ t j,t<c.T → j<c.V →
      initial.pids 2*(c.T*c.V)+t*c.V+j<bounds "v") :
    Safety.traceSafeList (engine (withDot M D)) bounds (suffix c seed final) current := by
  have ha := ctx.carry
  simp only [HasCarry] at ha
  generalize hbackend : engine (withDot M D)=E
  change Safety.traceSafeList E bounds [_,_,_,_] current
  by_cases last : n=c.chunks-1 ∧ c.T%64≠0
  all_goals
    simp [AccessSafety.singleton_safe,AccessSafety.pair_safe,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,numericEq,natLe,numericLe,numericLt,natLt,bop,
      ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
      Broadcast.leftIndex,Broadcast.rightIndex,keyPointer,valuePointer,
      ctx.pk,ctx.pv,ctx.it,ctx.context.bb,ctx.context.oi,ctx.context.db hn,ctx.context.di hn,ha,last]
  all_goals
    constructor
    · intro e t he ht
      exact hk _ _ he ht
    · intro t j ht hj
      exact hv _ _ ht hj

#axiomsClean suffix_safe

end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardHSource
