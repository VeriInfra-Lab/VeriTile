import bench.repairs.tritonbench_g.chunk_retention.Proofs.PureAccessSafety
import bench.repairs.tritonbench_g.chunk_retention.Proofs.ForwardOComplete
import bench.repairs.tritonbench_g.chunk_retention.Proofs.AccessSafetyLoops

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false
attribute [local simp 2000] AccessSafety.two_offset_pointer

private theorem entry_safe (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (scale : ℝ) (s : State α) (bounds : RegionName → Nat) :
    Safety.traceSafeList (engine (withDot M D)) bounds (parts c scale).before s := by
  apply AccessSafety.list_safe_of_steps
  intro statement member current
  change statement ∈ ([_,_,_,_,_,_,_,_,_,_,_] : List ComputeStmt) at member
  simp only [List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
  all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold]

private theorem body_safe (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (scale : ℝ) (initial current : State α) (f : Nat) (a : Accum c α)
    (ctx : Context M c initial current) (ha : HasAcc c a.1 a.2 current)
    (bounds : RegionName → Nat)
    (hq : ∀ t e,t<c.T → e<c.K → initial.pids 2*(c.T*c.K)+t*c.K+e<bounds "q")
    (hk : ∀ e t,e<c.K → t<c.T → initial.pids 2*(c.T*c.K)+e+t*c.K<bounds "k")
    (hh : ∀ e j,e<c.K → j<c.V →
      initial.pids 2*(c.chunks*c.K*c.V)+initial.pids 1*c.K*c.V+e*c.V+j<bounds "h") :
    Safety.traceSafeList (engine (withDot M D)) bounds (parts c scale).body
      (current.setReg "i_k" .nat [] (fun _ => f)) := by
  simp only [HasAcc] at ha
  simp [parts,kernel,Kernels.forwardO,ComputeKernel.surfaceBody,Source.dynamicRange?,
      Source.prefixBefore,Option.map,Option.getD,AccessSafety.singleton_safe,AccessSafety.pair_safe,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,numericEq,natLe,numericLe,numericLt,natLt,bop,
      ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
      Broadcast.leftIndex,Broadcast.rightIndex,ctx.memory,ctx.iv,ctx.it,ctx.bh,ctx.di,ctx.ds,ha.1,ha.2]
  refine ⟨?_,?_,?_⟩
  · intro t e ht he
    exact hq _ _ ht he
  · intro e t he ht
    exact hk _ _ he ht
  · intro e j he hj
    exact hh _ _ he hj

private theorem finish_safe (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (scale : ℝ) (initial current : State α) (ctx : Context M c initial current)
    (ha : HasAcc c (accumulated M D c initial).1 (accumulated M D c initial).2 current)
    (bounds : RegionName → Nat)
    (hv : ∀ i,active c initial i → outputAddress c initial i<bounds "v")
    (ho : ∀ i,active c initial i → outputAddress c initial i<bounds "o") :
    Safety.traceSafeList (engine (withDot M D)) bounds (parts c scale).after current := by
  simp only [HasAcc] at ha
  simp [parts,kernel,Kernels.forwardO,ComputeKernel.surfaceBody,Source.dynamicRange?,
      Source.prefixBefore,Option.map,Option.getD,AccessSafety.singleton_safe,AccessSafety.pair_safe,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,numericEq,natLe,numericLe,numericLt,natLt,bop,
      ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
      Broadcast.leftIndex,Broadcast.rightIndex,ctx.memory,ctx.iv,ctx.it,ctx.bh,ctx.di,ctx.ds,ha.1,ha.2,FP.Structural.store]
  constructor
  · intro t j ht hj
    exact hv (t,j,PUnit.unit) ⟨ht,hj⟩
  · intro t j ht hj
    exact ho (t,j,PUnit.unit) ⟨ht,hj⟩

/-- Actual forward-output loads, feature-block loop, and masked output store. -/
theorem source_safe (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (scale : ℝ) (initial : State α) (bounds : RegionName → Nat)
    (hq : ∀ t e,t<c.T → e<c.K → initial.pids 2*(c.T*c.K)+t*c.K+e<bounds "q")
    (hk : ∀ e t,e<c.K → t<c.T → initial.pids 2*(c.T*c.K)+e+t*c.K<bounds "k")
    (hh : ∀ e j,e<c.K → j<c.V →
      initial.pids 2*(c.chunks*c.K*c.V)+initial.pids 1*c.K*c.V+e*c.V+j<bounds "h")
    (hv : ∀ i,active c initial i → outputAddress c initial i<bounds "v")
    (ho : ∀ i,active c initial i → outputAddress c initial i<bounds "o") :
    TraceSafe (engine (withDot M D)) bounds (kernel c scale) initial := by
  let start := entryExecution M D c scale initial
  have hc := entryContext M D c scale initial
  have safeLoop : Safety.rangeSafe (engine (withDot M D)) bounds "i_k" 0 c.keyBlocks 1
      (parts c scale).body start.val := by
    let I := fun (_ : Nat) s => Context M c initial s ∧ ∃ a : Accum c α,HasAcc c a.1 a.2 s
    apply AccessSafety.range_safe _ bounds "i_k" c.keyBlocks _ I _ 0 start.val ⟨hc.1,(fun _=>zero M,fun _=>zero M),hc.2⟩
    intro f current hf h
    obtain ⟨a,ha⟩ := h.2
    let next := bodyExecution M D c scale initial current f a.1 a.2 h.1 ha
    have hn := bodyContext M D c scale initial current f a h.1 ha
    exact ⟨body_safe M D c scale initial current f a h.1 ha bounds hq hk hh,
      next.val,next.property,hn.1,_,hn.2⟩
  obtain ⟨mid,hm,ctx,ha⟩ := loopExecution M D c scale initial start.val hc.1 hc.2
  have h0' : (parts c scale).start=.constNat 0 := rfl
  have hn' : (parts c scale).stop=.div .nat .nil
      (.sub .nat .nil (.add .nat .nil (.constNat c.K) (.constNat c.BK)) (.constNat 1))
      (.constNat c.BK) := rfl
  have h1' : (parts c scale).stride=.constNat 1 := rfl
  unfold TraceSafe
  rw [source_parts,Safety.traceSafeList_append,Safety.traceSafeList_append]
  refine ⟨⟨entry_safe M D c scale initial bounds,?_⟩,?_⟩
  · intro t ran
    rw [start.property] at ran
    cases ran
    apply Safety.traceSafeList_cons_intro
    · simpa [Safety.stepSafe_unfold,Safety.safeOp_unfold,h0',hn',h1',evalOp_unfold,
        numeric,bop,Broadcast.leftIndex,Broadcast.rightIndex,Shape.keyBlocks] using safeLoop
    · intro _ _; exact Safety.traceSafeList_nil _ _ _
  · intro t ran
    rw [run_append,start.property] at ran
    simp only [Option.bind_some,run,step,h0',hn',h1',evalOp_unfold,numeric,bop,
      Broadcast.leftIndex,Broadcast.rightIndex] at ran
    change (range (engine (withDot M D)) "i_k" 0 c.keyBlocks 1 (parts c scale).body start.val).bind
      (fun s => some s)=some t at ran
    rw [hm] at ran
    simp only [Option.bind_some] at ran
    cases ran
    exact finish_safe M D c scale initial mid ctx ha bounds hv ho

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource
