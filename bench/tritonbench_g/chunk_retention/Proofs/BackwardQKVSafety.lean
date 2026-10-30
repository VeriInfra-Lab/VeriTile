import bench.tritonbench_g.chunk_retention.Sources.BackwardQKVComplete
import bench.tritonbench_g.chunk_retention.Proofs.AccessSafetyLoops
import bench.tritonbench_g.chunk_retention.Proofs.PureAccessSafety

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.BackwardQKVSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false
attribute [local simp 2000] AccessSafety.two_offset_pointer

private theorem entry_safe (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (scale : ℝ) (s : State α) (bounds : RegionName→Nat)
    (hq : ∀ e t,e<c.K → t<c.T → s.pids 2*(c.T*c.K)+e+t*c.K<bounds "q")
    (hk : ∀ t e,t<c.T → e<c.K → s.pids 2*(c.T*c.K)+t*c.K+e<bounds "k") :
    Safety.traceSafeList (engine (withDot M D)) bounds (parts c scale).before s := by
  by_cases last : s.pids 1=c.chunks-1 ∧ c.T%64≠0
  all_goals
    simp [parts,kernel,Kernels.backwardQKV,ComputeKernel.surfaceBody,Source.dynamicRange?,
      Source.prefixBefore,Option.map,Option.getD,AccessSafety.singleton_safe,AccessSafety.pair_safe,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,numericEq,natLe,numericLe,numericLt,natLt,bop,
      ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
      Broadcast.leftIndex,Broadcast.rightIndex,last]
  all_goals
    constructor
    · intro e t he ht; exact hq _ _ he ht
    · intro t e ht he; exact hk _ _ ht he

private theorem body_safe (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (initial current : State α) (f : Nat) (a : Accum c α)
    (ctx : Context M D c scale initial current) (ha : HasAcc c a current) (bounds : RegionName→Nat)
    (hv : ∀ t j,t<c.T → j<c.V → initial.pids 2*(c.T*c.V)+t*c.V+j<bounds "v")
    (hg : ∀ t j,t<c.T → j<c.V → initial.pids 2*(c.T*c.V)+t*c.V+j<bounds "incoming")
    (hh : ∀ j e,j<c.V → e<c.K →
      initial.pids 2*(c.chunks*c.K*c.V)+initial.pids 1*c.K*c.V+j+e*c.V<bounds "h")
    (hdh : ∀ e j,e<c.K → j<c.V →
      initial.pids 2*(c.chunks*c.K*c.V)+initial.pids 1*c.K*c.V+e*c.V+j<bounds "dh")
    (hdv : ∀ i,valueActive c initial f i → partialAddress c initial f i<bounds "dv") :
    Safety.traceSafeList (engine (withDot M D)) bounds (parts c scale).body
      (current.setReg "i_v" .nat [] (fun _=>f)) := by
  simp only [HasAcc] at ha
  simp [parts,kernel,Kernels.backwardQKV,ComputeKernel.surfaceBody,Source.dynamicRange?,
      Source.prefixBefore,Option.map,Option.getD,AccessSafety.singleton_safe,AccessSafety.pair_safe,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,numericEq,natLe,numericLe,numericLt,natLt,bop,
      ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
      Broadcast.leftIndex,Broadcast.rightIndex,ctx.ik,ctx.it,ctx.bh,ctx.nh,ctx.dq,ctx.dk,ctx.ds,ctx.query,ctx.key,ctx.score,ha.1,ha.2.1,ha.2.2,FP.Structural.store]
  refine ⟨?_,?_,?_,?_,?_⟩
  · intro t j ht hj; exact hv _ _ ht hj
  · intro t j ht hj; exact hg _ _ ht hj
  · intro j e hj he; exact hh _ _ hj he
  · intro e j he hj; exact hdh _ _ he hj
  · intro t j ht hj; exact hdv (t,j,PUnit.unit) ⟨ht,hj⟩

private theorem finish_safe (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (initial current : State α) (a : Accum c α)
    (ctx : Context M D c scale initial current) (ha : HasAcc c a current) (bounds : RegionName→Nat)
    (hq : ∀ i,keyActive c initial i → keyAddress c initial i<bounds "dq")
    (hk : ∀ i,keyActive c initial i → keyAddress c initial i<bounds "dk") :
    Safety.traceSafeList (engine (withDot M D)) bounds (parts c scale).after current := by
  simp only [HasAcc] at ha
  simp [parts,kernel,Kernels.backwardQKV,ComputeKernel.surfaceBody,Source.dynamicRange?,
      Source.prefixBefore,Option.map,Option.getD,AccessSafety.singleton_safe,AccessSafety.pair_safe,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,numericEq,natLe,numericLe,numericLt,natLt,bop,
      ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
      Broadcast.leftIndex,Broadcast.rightIndex,ctx.ik,ctx.it,ctx.bh,ctx.dq,ctx.dk,ctx.ds,ctx.query,ctx.key,ha.1,ha.2.1,ha.2.2,FP.Structural.store]
  constructor
  · intro t e ht he; exact hq (t,e,PUnit.unit) ⟨ht,he⟩
  · intro t e ht he; exact hk (t,e,PUnit.unit) ⟨ht,he⟩

/-- Actual four-input value-block loop, every partial dV store, and complete
dQ/dK epilogue. Numeric dot values are supplied by the existing source model. -/
theorem source_safe (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (s : State α) (bounds : RegionName→Nat)
    (hq : ∀ e t,e<c.K → t<c.T → s.pids 2*(c.T*c.K)+e+t*c.K<bounds "q")
    (hk : ∀ t e,t<c.T → e<c.K → s.pids 2*(c.T*c.K)+t*c.K+e<bounds "k")
    (hv : ∀ t j,t<c.T → j<c.V → s.pids 2*(c.T*c.V)+t*c.V+j<bounds "v")
    (hg : ∀ t j,t<c.T → j<c.V → s.pids 2*(c.T*c.V)+t*c.V+j<bounds "incoming")
    (hh : ∀ j e,j<c.V → e<c.K →
      s.pids 2*(c.chunks*c.K*c.V)+s.pids 1*c.K*c.V+j+e*c.V<bounds "h")
    (hdh : ∀ e j,e<c.K → j<c.V →
      s.pids 2*(c.chunks*c.K*c.V)+s.pids 1*c.K*c.V+e*c.V+j<bounds "dh")
    (hdv : ∀ f,f<c.valueBlocks → ∀ i,valueActive c s f i → partialAddress c s f i<bounds "dv")
    (hdq : ∀ i,keyActive c s i → keyAddress c s i<bounds "dq")
    (hdk : ∀ i,keyActive c s i → keyAddress c s i<bounds "dk") :
    TraceSafe (engine (withDot M D)) bounds (kernel c scale) s := by
  let start := entryExecution M D c scale s
  have hc := entryContext M D c scale s
  have safeLoop : Safety.rangeSafe (engine (withDot M D)) bounds "i_v" 0 c.valueBlocks 1
      (parts c scale).body start.val := by
    let I := fun (_ : Nat) current=>Context M D c scale s current ∧ ∃ a,HasAcc c a current
    apply AccessSafety.range_safe _ bounds "i_v" c.valueBlocks _ I _ 0 start.val
      ⟨hc.1,⟨fun _=>zero M,fun _=>zero M,fun _=>zero M⟩,hc.2⟩
    intro f current hf h
    obtain ⟨a,ha⟩ := h.2
    let next := bodyExecution M D c scale s current f a h.1 ha
    exact ⟨body_safe M D c scale s current f a h.1 ha bounds hv hg hh hdh (hdv f hf),
      next.val,next.property,bodyContext M D c scale s current f a h.1 ha,
      _,bodyAcc M D c scale s current f a h.1 ha⟩
  obtain ⟨mid,hm,ctx,ha,hmem⟩ := loopExecution M D c scale s start.val hc.1 hc.2 (entryMemory M D c scale s)
  have h0' : (parts c scale).start=.constNat 0 := rfl
  have hn' : (parts c scale).stop=.div .nat .nil
      (.sub .nat .nil (.add .nat .nil (.constNat c.V) (.constNat c.BV)) (.constNat 1)) (.constNat c.BV) := rfl
  have h1' : (parts c scale).stride=.constNat 1 := rfl
  unfold TraceSafe
  rw [source_parts,Safety.traceSafeList_append,Safety.traceSafeList_append]
  refine ⟨⟨entry_safe M D c scale s bounds hq hk,?_⟩,?_⟩
  · intro t ran
    rw [start.property] at ran
    cases ran
    apply Safety.traceSafeList_cons_intro
    · simpa [Safety.stepSafe_unfold,Safety.safeOp_unfold,h0',hn',h1',evalOp_unfold,
        numeric,bop,Broadcast.leftIndex,Broadcast.rightIndex,Shape.valueBlocks] using safeLoop
    · intro _ _; exact Safety.traceSafeList_nil _ _ _
  · intro t ran
    rw [run_append,start.property] at ran
    simp only [Option.bind_some,run,step,h0',hn',h1',evalOp_unfold,numeric,bop,
      Broadcast.leftIndex,Broadcast.rightIndex] at ran
    change (range (engine (withDot M D)) "i_v" 0 c.valueBlocks 1 (parts c scale).body start.val).bind
      (fun s=>some s)=some t at ran
    rw [hm] at ran
    simp only [Option.bind_some] at ran
    cases ran
    exact finish_safe M D c scale s mid _ ctx ha bounds hdq hdk

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.BackwardQKVSource
