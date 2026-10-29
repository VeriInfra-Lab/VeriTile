import bench.repairs.tritonbench_g.chunk_delta_fwd_static_dot.Proofs.SourceSafetyEdges

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.SourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite Safety ChunkDeltaFwd
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 24000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem inner_body (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (pid : PID) (input : Memory α) (current : State α) (c b : Nat)
    (frozen acc : HValues p α) (ctx : Ready pid input current)
    (hh : HasH p frozen current) (ha : HasAcc p acc current)
    (ht : current.regs .nat [] "i_t"=some (fun _ => c))
    (bounds : RegionName → Nat) (hb : Bounds p pid bounds) :
    traceSafeList (engine (withDot M D)) bounds (inner p).body
      (current.setReg "i_c" .nat [] (fun _ => b)) := by
  simp [ChunkDeltaFwd.inner,outer,kernel,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD]
  simp only [HasH] at hh
  simp only [HasAcc] at ha
  simp [
    singleton,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
    engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,FP.Structural.store,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
    ctx.ik,ctx.iv,ctx.bh,hh,ha,ht]
  refine ⟨?_,?_,?_,?_⟩
  · intro k t hk ht
    exact hb.key (c*p.BT+b*p.BC+t.val) k.val ht hk
  · intro t k ht hk
    exact hb.weight (c*p.BT+b*p.BC+t.val) k.val ht hk
  · intro t v ht hv
    exact hb.value (c*p.BT+b*p.BC+t.val) (pid.1*p.BV+v.val) ht hv
  · intro t v ht hv
    exact hb.corrected (c*p.BT+b*p.BC+t.val) (pid.1*p.BV+v.val) ht hv

theorem inner_range (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (pid : PID) (input : Memory α) (current : State α) (c : Nat)
    (frozen : HValues p α) (ctx : Ready pid input current)
    (hh : HasH p frozen current) (ha : HasAcc p (fun _ => zero M) current)
    (ht : current.regs .nat [] "i_t"=some (fun _ => c))
    (bounds : RegionName → Nat) (hb : Bounds p pid bounds) :
    rangeSafe (engine (withDot M D)) bounds "i_c" 0 (innerCount p) 1
      (inner p).body current := by
  let ap := fun n => foldRange (fun b acc => accumulatedStep M D p pid input c b frozen acc)
    0 n 1 (fun _ => zero M)
  let I := fun n (s : State α) => Ready pid input s ∧ HasH p frozen s ∧
    HasAcc p (ap n) s ∧ s.regs .nat [] "i_t"=some (fun _ => c)
  apply rangeSafe_of_invariant (engine (withDot M D)) bounds "i_c" (innerCount p) 1
    (inner p).body I
  · intro n _ s inv
    exact inner_body M D p pid input s c n frozen (ap n)
      inv.1 inv.2.1 inv.2.2.1 inv.2.2.2 bounds hb
  · intro n _ s inv t ran
    let e := innerExecution M D p pid input s c n frozen (ap n)
      inv.1 inv.2.1 inv.2.2.1 inv.2.2.2
    have eq : t = e.val := Option.some.inj (ran.symm.trans e.property)
    subst t
    have hv := inner_values M D p pid input s c n frozen (ap n)
      inv.1 inv.2.1 inv.2.2.1 inv.2.2.2
    refine ⟨hv.1,hv.2.1,?_,hv.2.2.2⟩
    simpa only [ap,foldRange_succ] using hv.2.2.1
  · exact ⟨ctx,hh,by simpa only [ap,foldRange_zero] using ha,ht⟩

end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.SourceSafety
