import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeSafetyMerge

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite FPMerge
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem merge_finish (M : Algebra α) (p : Layout) (bounds : RegionName → Nat)
    (initial current : State α) (ctx : Context p initial current)
    (capacity : MergeCapacity p initial bounds) :
    Safety.traceSafeList (engine M) bounds (parts p).after current := by
  simp [parts,branch,kernel,TypedMerge.original,ComputeKernel.surfaceBody,
    Source.conditional?,Source.dynamicRange?,Source.prefixBefore]
  simp only [Safety.traceSafeList_cons,singleton]
  simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,safeBlockPtr2]
  simp [Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    ctx.it,ctx.ic,ctx.bh,ctx.nh,BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,
    TileShape.indexToList]
  intro i j row column
  exact capacity.output (i,j,PUnit.unit) ⟨row,column⟩

theorem merge_range (M : Algebra α) (p : Layout) (bounds : RegionName → Nat)
    (initial current : State α) (ctx : Context p initial current) (a : Index p → α) (ha : HasAcc p a current)
    (capacity : MergeCapacity p initial bounds) :
    Safety.rangeSafe (engine M) bounds "i_k" 0 p.NK 1 (parts p).body current := by
  apply Safety.rangeSafe_of_invariant (engine M) bounds "i_k" p.NK 1 (parts p).body
    (fun _ t => ∃ a,Context p initial t ∧ HasAcc p a t)
  · intro f active t h
    obtain ⟨a,ctx,ha⟩ := h
    exact merge_body M p bounds initial t f active ctx capacity
  · intro f active t h u ran
    obtain ⟨a,ctx,ha⟩ := h
    have equal := Option.some.inj (ran.symm.trans (bodyExecution M p initial t f a ctx ha).property)
    subst u
    exact ⟨update M p initial f a,bodyContext M p initial t f a ctx ha⟩
  · exact ⟨a,ctx,ha⟩

theorem merge_source (M : Algebra α) (p : Layout) (bounds : RegionName → Nat)
    (s : State α) (capacity : MergeCapacity p s bounds) :
    TraceSafe (engine M) bounds (kernel p) s := by
  let start := entryExecution M p s
  have hs : run (engine M) (parts p).before (prologue M p s).val=some start.val := by
    simpa only [run_append,(prologue M p s).property,Option.bind_some] using start.property
  change Safety.traceSafeList (engine M) bounds (kernel p).surfaceBody s
  rw [source_parts,Safety.traceSafeList_append]
  refine ⟨merge_prologue M p bounds s,?_⟩
  intro t ran
  have same := Option.some.inj (ran.symm.trans (prologue M p s).property)
  subst t
  rw [singleton,Safety.stepSafe_unfold]
  constructor
  · simp [branch,kernel,TypedMerge.original,ComputeKernel.surfaceBody,
      Source.conditional?,Source.prefixBefore,Safety.safeOp_unfold]
  · rw [condition_eval]
    by_cases active : enabled p s
    · simp only [active,decide_true,ite_true]
      rw [List.append_assoc,Safety.traceSafeList_append]
      refine ⟨merge_entry M p bounds _,?_⟩
      intro t ran
      rw [ran] at hs
      injection hs with same
      subst t
      obtain ⟨mid,executed,ctx,acc⟩ := loopExecution M p s start.val
        (entryContext M p s).1 (entryContext M p s).2
      have h0 : (parts p).start = .constNat 0 := rfl
      have hn : (parts p).stop = .constNat p.NK := rfl
      have h1 : (parts p).stride = .constNat 1 := rfl
      have hl : step (engine M) (.forRangeDyn "i_k" (parts p).start (parts p).stop
          (parts p).stride (parts p).body) start.val=some mid := by
        simpa only [step,h0,hn,h1,evalOp_unfold,Option.bind,Option.bind_some] using executed
      simp only [List.singleton_append]
      apply Safety.traceSafeList_cons_intro
      · simp only [Safety.stepSafe_unfold,h0,hn,h1,Safety.safeOp_unfold,evalOp_unfold,
          Option.bind,Option.bind_some,true_and]
        exact merge_range M p bounds s start.val (entryContext M p s).1 _
          (entryContext M p s).2 capacity
      · intro t ran
        rw [ran] at hl
        injection hl with same
        subst t
        exact merge_finish M p bounds s mid ctx capacity
    · simp [active]

#axiomsClean merge_finish
#axiomsClean merge_source
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeSafety
