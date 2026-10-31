import bench.tritonbench_g.chunk_retention.Sources.BackwardHComplete
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
    simp [parts,kernel,Kernels.backwardH,ComputeKernel.surfaceBody,Source.dynamicRange?,
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
    (ctx : Context M c initial current) (ha : HasCarry c carry current) (bounds : RegionName→Nat)
    (hq : ∀ e t,e<c.K → t<c.T → initial.pids 2*(c.T*c.K)+e+t*c.K<bounds "q")
    (hg : ∀ t j,t<c.T → j<c.V → initial.pids 2*(c.T*c.V)+t*c.V+j<bounds "incoming")
    (hh : ∀ i,active c initial i → historyAddress c initial (chunk c n) i<bounds "dh") :
    Safety.traceSafeList (engine (withDot M D)) bounds (parts c scale terminal).body
      (current.setReg "j" .nat [] (fun _=>n)) := by
  simp only [HasCarry] at ha
  by_cases atEnd : c.chunks-1-n=c.chunks-1
  all_goals by_cases rem : c.T%64=0
  all_goals
    simp [parts,kernel,Kernels.backwardH,ComputeKernel.surfaceBody,Source.dynamicRange?,
      Source.prefixBefore,Option.map,Option.getD,AccessSafety.singleton_safe,AccessSafety.pair_safe,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,KernelModel.signed_head_mod,Int.toNat_mul,Int.toNat_add,Int.toNat_natCast,numericEq,natLe,numericLe,numericLt,natLt,bop,
      ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
      Broadcast.leftIndex,Broadcast.rightIndex,ctx.ik,ctx.iv,ctx.bh,KernelModel.signed_head_mod,Int.toNat_mul,Int.toNat_add,Int.toNat_natCast,ctx.bb,ctx.di,ha,atEnd,rem,FP.Structural.store]
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
  have safeLoop : Safety.rangeSafe (engine (withDot M D)) bounds "j" 0 c.chunks 1
      (parts c scale terminal).body start.val := by
    let I := fun (_ : Nat) current=>Context M c s current ∧ ∃ carry,HasCarry c carry current
    apply AccessSafety.range_safe _ bounds "j" c.chunks _ I _ 0 start.val ⟨hc.1,_,hc.2⟩
    intro n current hn h
    obtain ⟨carry,ha⟩ := h.2
    let next := bodyExecution M D c scale terminal s current n carry h.1 ha
    exact ⟨body_safe M D c scale terminal s current n carry h.1 ha bounds hq hg (hh n hn),
      next.val,next.property,bodyContext M D c scale terminal s current n carry h.1 ha,
      _,bodyCarry M D c scale terminal s current n carry h.1 ha⟩
  unfold TraceSafe
  rw [source_parts,Safety.traceSafeList_append]
  refine ⟨entry_safe M D c scale terminal s bounds ht,?_⟩
  intro current ran
  rw [start.property] at ran
  cases ran
  simpa only [AccessSafety.singleton_safe,Safety.stepSafe_unfold] using safeLoop

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.BackwardHSource
