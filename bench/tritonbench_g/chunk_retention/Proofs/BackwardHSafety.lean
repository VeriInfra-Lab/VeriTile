import bench.tritonbench_g.chunk_retention.Proofs.Sources.BackwardHComplete
import bench.tritonbench_g.chunk_linear_attn.Proofs.SignedRangeSafety
import bench.tritonbench_g.chunk_retention.Proofs.AccessSafetyLoops
import bench.tritonbench_g.chunk_retention.Proofs.PureAccessSafety

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.BackwardHSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false
attribute [local simp 2000] AccessSafety.two_offset_pointer

private theorem entry_safe (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (scale : ℝ) (terminal : Bool) (s : State α) (bounds : RegionName→Nat)
    (ht : terminal=Bool.true → ∀ i,active c s i → stateAddress c s i<bounds "terminal") :
    Safety.traceSafeList (engine (withDot M D)) bounds (parts c scale terminal).before s := by
  cases terminal
  all_goals
    simp [parts,intParts,indexState,kernel,Kernels.backwardHKernel,ComputeKernel.surfaceBody,Source.dynamicRange?,Source.intRange?,
      Source.prefixBefore,Option.map,Option.getD,AccessSafety.singleton_safe,AccessSafety.pair_safe,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,KernelModel.signed_head_mod,Int.toNat_mul,Int.toNat_add,Int.toNat_natCast,natLe,numericLe,numericLt,natLt,bop,
      ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  intro e j he hj
  exact ht rfl (e,j,PUnit.unit) ⟨he,hj⟩

private theorem body_safe (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (scale : ℝ) (terminal : Bool) (initial current : State α) (n : Nat) (carry : Index c→α)
    (ctx : Context M c initial current) (ha : HasCarry c carry current) (hn : n<c.chunks) (bounds : RegionName→Nat)
    (hq : ∀ e t,e<c.K → t<c.T → initial.pids 2*(c.T*c.K)+e+t*c.K<bounds "q")
    (hg : ∀ t j,t<c.T → j<c.V → initial.pids 2*(c.T*c.V)+t*c.V+j<bounds "incoming")
    (hh : ∀ i,active c initial i → historyAddress c initial (chunk c n) i<bounds "dh") :
    Safety.traceSafeList (engine (withDot M D)) bounds (parts c scale terminal).body
      (indexState c n current) := by
  simp only [HasCarry] at ha
  have clock : (0 : Int)-((n : Int)+1-(c.chunks : Int))=((c.chunks-1-n : Nat) : Int) := by omega
  have lastClock : (c.chunks : Int)-1=((c.chunks-1 : Nat) : Int) := by omega
  by_cases atEnd : c.chunks-1-n=c.chunks-1
  all_goals by_cases rem : c.T%64=0
  all_goals
    simp [parts,intParts,indexState,kernel,Kernels.backwardHKernel,ComputeKernel.surfaceBody,Source.dynamicRange?,Source.intRange?,
      Source.prefixBefore,Option.map,Option.getD,AccessSafety.singleton_safe,AccessSafety.pair_safe,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,KernelModel.signed_head_mod,Int.toNat_mul,Int.toNat_add,Int.toNat_natCast,numericEq,natLe,numericLe,numericLt,natLt,bop,
      ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
      Broadcast.leftIndex,Broadcast.rightIndex,ctx.ik,ctx.iv,ctx.bh,KernelModel.signed_head_mod,Int.toNat_mul,Int.toNat_add,Int.toNat_natCast,ctx.bb,ctx.di,ha,atEnd,rem,clock,lastClock,FP.Structural.store]
  all_goals
    refine ⟨?_,?_,?_⟩
    · intro e j he hj
      simpa only [historyAddress,ForwardHSource.historyAddress,feature,column,
        ForwardHSource.feature,ForwardHSource.column,chunk,atEnd] using hh (e,j,PUnit.unit) ⟨he,hj⟩
    · intro t j ht hj; exact hg _ _ ht hj
    · intro e t he ht; exact hq _ _ he ht

/-- Reverse-history terminal loads, all history stores, both tail-masked
input loads and the complete reverse-count loop follow the actual source. -/
theorem source_safe (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (scale : ℝ) (terminal : Bool) (s : State α) (bounds : RegionName→Nat)
    (ht : terminal=Bool.true → ∀ i,active c s i → stateAddress c s i<bounds "terminal")
    (hq : ∀ e t,e<c.K → t<c.T → s.pids 2*(c.T*c.K)+e+t*c.K<bounds "q")
    (hg : ∀ t j,t<c.T → j<c.V → s.pids 2*(c.T*c.V)+t*c.V+j<bounds "incoming")
    (hh : ∀ n,n<c.chunks → ∀ i,active c s i → historyAddress c s (chunk c n) i<bounds "dh") :
    TraceSafe (engine (withDot M D)) bounds (kernel c scale terminal) s := by
  let start := entryExecution M D c scale terminal s
  have hc := entryContext M D c scale terminal s
  have safeLoop : Safety.stepSafe (engine (withDot M D)) bounds (loopStmt c scale terminal) start.val := by
    let I := fun (_ : Nat) current=>Context M c s current ∧ ∃ carry,HasCarry c carry current
    have bodySafe : ∀ n, n<c.chunks → ∀ t, I n t →
        Safety.traceSafeList (engine (withDot M D)) bounds (parts c scale terminal).body
          (t.setReg "__rev_i_t" .int [] (fun _ => (1-(c.chunks : Int))+(n : Int))) := by
      intro n hn t ⟨ctx,carry,ha⟩
      simpa [indexState,add_comm,add_left_comm,add_assoc,sub_eq_add_neg] using
        body_safe M D c scale terminal s t n carry ctx ha hn bounds hq hg (hh n hn)
    have preserved : ∀ n, n<c.chunks → ∀ t, I n t → ∀ u,
        run (engine (withDot M D)) (parts c scale terminal).body
          (t.setReg "__rev_i_t" .int [] (fun _ => (1-(c.chunks : Int))+(n : Int)))=some u → I (n+1) u := by
      intro n hn t ⟨ctx,carry,ha⟩ u ran
      let next := bodyExecution M D c scale terminal s t n carry ctx ha hn
      have executed : run (engine (withDot M D)) (parts c scale terminal).body
          (indexState c n t)=some u := by
        simpa [indexState,add_comm,add_left_comm,add_assoc,sub_eq_add_neg] using ran
      have same : next.val=u := Option.some.inj (next.property.symm.trans executed)
      subst u
      exact ⟨bodyContext M D c scale terminal s t n carry ctx ha hn,
        _,bodyCarry M D c scale terminal s t n carry ctx ha hn⟩
    have safe := VeriTile.Bench.Repairs.ChunkLinearAttn.SignedRange.clock_safe
      (engine (withDot M D)) bounds "__rev_i_t" (parts c scale terminal).body
      (1-(c.chunks : Int)) c.chunks I bodySafe preserved 0 (Nat.zero_le _) start.val ⟨hc.1,_,hc.2⟩
    have first : (intParts c scale terminal).start =
        .sub .int .nil (.constInt 0) (.sub .int .nil (.constInt (Int.ofNat c.chunks)) (.constInt 1)) := rfl
    have last : (intParts c scale terminal).stop = .constInt 1 := rfl
    have stride : (intParts c scale terminal).stride = .constNat 1 := rfl
    simp only [loopStmt,Safety.stepSafe_unfold,first,last,stride,Safety.safeOp_unfold,
      evalOp_unfold,numeric,bop,Option.bind,Option.bind_some,true_and,and_true]
    change Safety.rangeIntSafe (engine (withDot M D)) bounds "__rev_i_t"
      (0-((c.chunks : Int)-1)) 1 1 (intParts c scale terminal).body start.val
    have beginEq : (0 : Int)-((c.chunks : Int)-1)=(1-(c.chunks : Int))+(0 : Nat) := by omega
    have endEq : (1-(c.chunks : Int))+(c.chunks : Int)=1 := by omega
    rw [beginEq]
    simpa only [endEq] using safe
  unfold TraceSafe
  rw [source_parts,Safety.traceSafeList_append]
  refine ⟨entry_safe M D c scale terminal s bounds ht,?_⟩
  intro current ran
  rw [start.property] at ran
  cases ran
  simpa only [AccessSafety.singleton_safe] using safeLoop

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.BackwardHSource
