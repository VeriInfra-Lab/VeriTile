import bench.tritonbench_g.chunk_retention_ops.Proofs.Sources.KernelBackwardProgram
import bench.tritonbench_g.chunk_retention.Proofs.PureAccessSafety
import bench.tritonbench_g.chunk_retention.Proofs.AccessSafetyLoops
import VeriTile.Triton.Float.StructuralSafetyLoops
import bench.tritonbench_g.chunk_linear_attn.Proofs.SignedRangeSafety

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
    simp [parts,intParts,indexState,kernel,Source.backwardHKernel,ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.intRange?,
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
    (hc : HasCarry p carry s) (hn : n<p.NT) (bounds : RegionName→Nat)
    (hq : ∀ e t,e<p.K → t<p.T → pid.z*(p.T*p.K)+e+t*p.K<bounds "q")
    (hg : ∀ t j,t<p.T → j<p.V → pid.z*(p.T*p.V)+t*p.V+j<bounds "incoming")
    (hh : ∀ i,hActive p pid i → hOffset p pid (time p n) i<bounds "dhistory") :
    Safety.traceSafeList (engine (withDot M D)) bounds (parts p scale terminal initial).body
      (indexState p n s) := by
  simp only [HasCarry] at hc
  have clock : (p.NT : Int)-((n : Int)+1)=((p.NT-1-n : Nat) : Int) := by omega
  simp [parts,intParts,indexState,kernel,Source.backwardHKernel,ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.intRange?,
      Triton.Source.prefixBefore,Option.map,Option.getD,AccessSafety.singleton_safe,AccessSafety.pair_safe,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,numericEq,natLe,numericLe,numericLt,natLt,bop,
      ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
      Broadcast.leftIndex,Broadcast.rightIndex,ctx.ik,ctx.iv,ctx.bh,ctx.rate,ctx.lane,ctx.weights,hc,FP.Structural.store,clock,natural_history_address,natural_offset]
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
    simp [parts,intParts,indexState,kernel,Source.backwardHKernel,ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.intRange?,
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
  have safeLoop : Safety.stepSafe (engine (withDot M D)) bounds (loopStmt p scale terminal initial) start.val := by
    have bodySafe : ∀ n, n<p.NT → ∀ t, I n t →
        Safety.traceSafeList (engine (withDot M D)) bounds (parts p scale terminal initial).body
          (t.setReg "__rev_i_t" .int [] (fun _ => (1-(p.NT : Int))+(n : Int))) := by
      intro n hn t inv
      obtain ⟨v,hv⟩ := inv.2
      simpa [indexState,add_comm,add_left_comm,add_assoc,sub_eq_add_neg] using
        body_safe M D p scale terminal initial pid s.mem t n v inv.1 hv hn bounds hq hg (hh n hn)
    have preserved : ∀ n, n<p.NT → ∀ t, I n t → ∀ u,
        run (engine (withDot M D)) (parts p scale terminal initial).body
          (t.setReg "__rev_i_t" .int [] (fun _ => (1-(p.NT : Int))+(n : Int)))=some u → I (n+1) u := by
      intro n hn t inv u ran
      obtain ⟨v,hv⟩ := inv.2
      obtain ⟨actual,executed,ctx,carry,_⟩ := bodyStep M D p scale terminal initial pid s.mem t n v inv.1 hv hn
      have oldRun : run (engine (withDot M D)) (parts p scale terminal initial).body
          (indexState p n t)=some u := by
        simpa [indexState,add_comm,add_left_comm,add_assoc,sub_eq_add_neg] using ran
      have same := Option.some.inj (executed.symm.trans oldRun)
      subst u
      exact ⟨ctx,_,carry⟩
    have safe := VeriTile.Bench.Repairs.ChunkLinearAttn.SignedRange.clock_safe
      (engine (withDot M D)) bounds "__rev_i_t" (parts p scale terminal initial).body
      (1-(p.NT : Int)) p.NT I bodySafe preserved 0 (Nat.zero_le _) start.val ⟨ready.1,_,ready.2⟩
    have first : (intParts p scale terminal initial).start =
        .sub .int .nil (.constInt 0) (.sub .int .nil (.constInt (Int.ofNat p.NT)) (.constInt 1)) := rfl
    have last : (intParts p scale terminal initial).stop = .constInt 1 := rfl
    have stride : (intParts p scale terminal initial).stride = .constNat 1 := rfl
    simp only [loopStmt,Safety.stepSafe_unfold,first,last,stride,Safety.safeOp_unfold,
      evalOp_unfold,numeric,bop,Option.bind,Option.bind_some,true_and,and_true]
    change Safety.rangeIntSafe (engine (withDot M D)) bounds "__rev_i_t"
      (0-((p.NT : Int)-1)) 1 1 (intParts p scale terminal initial).body start.val
    have beginEq : (0 : Int)-((p.NT : Int)-1)=(1-(p.NT : Int))+(0 : Nat) := by omega
    have endEq : (1-(p.NT : Int))+(p.NT : Int)=1 := by omega
    rw [beginEq]
    simpa only [endEq] using safe
  obtain ⟨mid,hm,ctx,carry,_⟩ := loop M D p scale terminal initial pid s.mem start.val ready.1 ready.2
  have loopRun := hm
  unfold TraceSafe
  rw [source_parts,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨entry_safe M D p scale terminal initial s bounds ht,?_⟩
  intro t ran
  rw [start.property] at ran
  cases ran
  apply Safety.traceSafeList_cons_of_step
    safeLoop loopRun
  exact finish_safe M D p scale terminal initial pid s.mem mid _ ctx carry bounds hi

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelBackward
