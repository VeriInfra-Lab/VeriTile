import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelBackwardProgram
import bench.repairs.tritonbench_g.chunk_retention.Proofs.PureAccessSafety
import bench.repairs.tritonbench_g.chunk_retention.Proofs.AccessSafetyLoops
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelBackward
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.Repairs.ChunkLinearAttn (hActive hOffset fOffset)
open VeriTile.Bench.Repairs.ChunkRetention.Whole001
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false
attribute [local simp 2000] AccessSafety.two_offset_pointer

private theorem entry_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (scale : ℝ) (terminal initial : Bool) (s : State α) (bounds : RegionName→Nat)
    (ht : terminal=Bool.true → ∀ i,hActive p ⟨s.pids 0,s.pids 1,s.pids 2⟩ i →
      fOffset p ⟨s.pids 0,s.pids 1,s.pids 2⟩ i<bounds "terminal") :
    Safety.traceSafeList (engine (withDot M D)) bounds (parts p scale terminal initial).before s := by
  cases terminal
  all_goals
    simp [parts,kernel,Source.backwardH,ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,
      Triton.Source.prefixBefore,Option.map,Option.getD,AccessSafety.singleton_safe,AccessSafety.pair_safe,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,
      ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  intro e j he hj
  exact ht rfl (e,j,PUnit.unit) ⟨he,hj⟩

private theorem body_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (scale : ℝ) (terminal initial : Bool) (pid : PID) (input : Memory α)
    (s : State α) (n : Nat) (carry : Carry p α) (ctx : Ready M p pid input s)
    (hc : HasCarry p carry s) (bounds : RegionName→Nat)
    (hq : ∀ e t,e<p.K → t<p.T → pid.z*(p.T*p.K)+e+t*p.K<bounds "q")
    (hg : ∀ t j,t<p.T → j<p.V → pid.z*(p.T*p.V)+t*p.V+j<bounds "incoming")
    (hh : ∀ i,hActive p pid i → hOffset p pid (time p n) i<bounds "dhistory") :
    Safety.traceSafeList (engine (withDot M D)) bounds (parts p scale terminal initial).body
      (s.setReg "j" .nat [] (fun _=>n)) := by
  simp only [HasCarry] at hc
  simp [parts,kernel,Source.backwardH,ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,
      Triton.Source.prefixBefore,Option.map,Option.getD,AccessSafety.singleton_safe,AccessSafety.pair_safe,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,numericEq,natLe,numericLe,numericLt,natLt,bop,
      ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
      Broadcast.leftIndex,Broadcast.rightIndex,ctx.ik,ctx.iv,ctx.bh,ctx.rate,ctx.lane,ctx.weights,hc,FP.Structural.store]
  refine ⟨?_,?_,?_⟩
  · intro e j he hj
    exact hh (e,j,PUnit.unit) ⟨he,hj⟩
  · intro e t he ht
    exact hq _ _ he ht
  · intro t j ht hj
    exact hg _ _ ht hj

private theorem finish_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (scale : ℝ) (terminal initial : Bool) (pid : PID) (input : Memory α)
    (s : State α) (carry : Carry p α) (ctx : Ready M p pid input s)
    (hc : HasCarry p carry s) (bounds : RegionName→Nat)
    (hi : initial=Bool.true → ∀ i,hActive p pid i → fOffset p pid i<bounds "initialGradient") :
    Safety.traceSafeList (engine (withDot M D)) bounds (parts p scale terminal initial).after s := by
  simp only [HasCarry] at hc
  cases initial
  all_goals
    simp [parts,kernel,Source.backwardH,ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,
      Triton.Source.prefixBefore,Option.map,Option.getD,AccessSafety.singleton_safe,AccessSafety.pair_safe,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,
      ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
      Broadcast.leftIndex,Broadcast.rightIndex,ctx.ik,ctx.iv,ctx.bh,hc]
  intro e j he hj
  exact hi rfl (e,j,PUnit.unit) ⟨he,hj⟩

/-- The actual reverse-state source: optional terminal load, every reverse
history write/read/update, and the optional initial-gradient store. -/
theorem source_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (scale : ℝ) (terminal initial : Bool) (s : State α) (hBT : p.BT≤2^31)
    (bounds : RegionName→Nat)
    (ht : terminal=Bool.true → ∀ i,hActive p ⟨s.pids 0,s.pids 1,s.pids 2⟩ i →
      fOffset p ⟨s.pids 0,s.pids 1,s.pids 2⟩ i<bounds "terminal")
    (hq : ∀ e t,e<p.K → t<p.T → s.pids 2*(p.T*p.K)+e+t*p.K<bounds "q")
    (hg : ∀ t j,t<p.T → j<p.V → s.pids 2*(p.T*p.V)+t*p.V+j<bounds "incoming")
    (hh : ∀ n,n<p.NT → ∀ i,hActive p ⟨s.pids 0,s.pids 1,s.pids 2⟩ i →
      hOffset p ⟨s.pids 0,s.pids 1,s.pids 2⟩ (time p n) i<bounds "dhistory")
    (hi : initial=Bool.true → ∀ i,hActive p ⟨s.pids 0,s.pids 1,s.pids 2⟩ i →
      fOffset p ⟨s.pids 0,s.pids 1,s.pids 2⟩ i<bounds "initialGradient") :
    TraceSafe (engine (withDot M D)) bounds (kernel p scale terminal initial) s := by
  let pid : PID := ⟨s.pids 0,s.pids 1,s.pids 2⟩
  let start := entry M D p scale terminal initial s
  have ready := entry_ready M D p scale terminal initial s hBT
  let I := fun (_ : Nat) (t : State α) => Ready M p pid s.mem t ∧ ∃ v,HasCarry p v t
  have safeLoop : Safety.rangeSafe (engine (withDot M D)) bounds "j" 0 p.NT 1
      (parts p scale terminal initial).body start.val := by
    apply Safety.rangeSafe_of_invariant _ bounds "j" p.NT 1 _ I
    · intro n hn t inv
      obtain ⟨v,hv⟩ := inv.2
      exact body_safe M D p scale terminal initial pid s.mem t n v inv.1 hv bounds hq hg (hh n hn)
    · intro n _ t inv u ran
      obtain ⟨v,hv⟩ := inv.2
      obtain ⟨actual,executes,ctx,carry,_⟩ := bodyStep M D p scale terminal initial pid s.mem t n v inv.1 hv
      have same := Option.some.inj (executes.symm.trans ran)
      subst u
      exact ⟨ctx,_,carry⟩
    · exact ⟨ready.1,_,ready.2⟩
  obtain ⟨mid,hm,ctx,carry,_⟩ := loop M D p scale terminal initial pid s.mem start.val ready.1 ready.2
  have loopRun : step (engine (withDot M D)) (loopStmt p scale terminal initial) start.val=some mid := by
    simpa only [loopStmt,step] using hm
  unfold TraceSafe
  rw [source_parts,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨entry_safe M D p scale terminal initial s bounds ht,?_⟩
  intro t ran
  rw [start.property] at ran
  cases ran
  apply Safety.traceSafeList_cons_of_step
    (by simpa only [loopStmt,Safety.stepSafe_unfold] using safeLoop) loopRun
  exact finish_safe M D p scale terminal initial pid s.mem mid _ ctx carry bounds hi

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelBackward
