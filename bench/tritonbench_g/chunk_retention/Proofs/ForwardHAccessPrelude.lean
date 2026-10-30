import bench.tritonbench_g.chunk_retention.Sources.ForwardHComplete
import bench.tritonbench_g.chunk_retention.Proofs.AccessSafetyLoops
import bench.tritonbench_g.chunk_retention.Proofs.PureAccessSafety

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardHSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 800000
set_option linter.unusedSimpArgs false

theorem prelude_safe (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (seed final : Bool) (initial current : State α) (n : Nat) (carry : Index c → α)
    (ctx : Context M c initial current n) (ha : HasCarry c carry current)
    (bounds : RegionName → Nat)
    (hh : ∀ i,active c initial i → historyAddress c initial n i<bounds "h") :
    Safety.traceSafeList (engine (withDot M D)) bounds (prelude c seed final)
      (current.setReg "i_t" .nat [] (fun _ => n)) := by
  simp only [HasCarry] at ha
  generalize hbackend : engine (withDot M D)=E
  change Safety.traceSafeList E bounds [_,_,_,_] (current.setReg "i_t" .nat [] (fun _=>n))
  apply Safety.traceSafeList_cons_intro
  · simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold]
    rw [AccessSafety.two_offset_pointer]
    simp only [Safety.safeOp_unfold,and_self]
  · intro next executed
    simp [step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,Option.bind,ComputeDType.eraseDType,numeric,bop,State.setReg,ctx.ik,ctx.iv,ctx.bh] at executed
    cases executed
    apply Safety.traceSafeList_cons_intro
    · simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold]
      rw [AccessSafety.two_offset_pointer]
      simp only [Safety.safeOp_unfold,and_self]
    · intro next executed
      simp [step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,Option.bind,ComputeDType.eraseDType,numeric,bop,State.setReg,ctx.ik,ctx.iv,ctx.bh] at executed
      cases executed
      apply Safety.traceSafeList_cons_intro
      · simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold]
        rw [AccessSafety.two_offset_pointer]
        simp only [Safety.safeOp_unfold,and_self]
      · intro next executed
        simp [step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,Option.bind,ComputeDType.eraseDType,numeric,bop,State.setReg,ctx.ik,ctx.iv,ctx.bh] at executed
        cases executed
        simp [AccessSafety.singleton_safe,Safety.stepSafe_unfold,
          Safety.safeExpr_unfold,Safety.safeCompute_unfold,AccessSafety.two_offset_pointer,Safety.safeOp_unfold,Safety.safeAddress_unfold,
          Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
          evalOp_unfold,Option.bind,State.setReg,ctx.ik,ctx.iv,ctx.bh,
          BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
        intro e j he hj
        exact hh (e,j,PUnit.unit) ⟨he,hj⟩

#axiomsClean prelude_safe

end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardHSource
