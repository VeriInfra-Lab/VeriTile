import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeSafetyInter

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite FPInter
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem inter_finish (M : Algebra α) (D : FPOutput.DotPrimitive α) (c : ProducerGeometry.Shape) (scale : ℝ) (bounds : RegionName → Nat)
    (initial current : State α) (ctx : Context c initial current)
    (capacity : InterCapacity c initial bounds) :
    Safety.traceSafeList (engine (FPOutput.withDot M D)) bounds (parts c scale).after current := by
  simp [parts,FPInter.inner,outer,kernel,SourceBridge.interNew,TypedInter.original,TypedInter.source,ProducerGeometry.layout,ProducerGeometry.regions,ComputeKernel.surfaceBody,
    Source.conditional?,Source.dynamicRange?,Source.prefixBefore]
  simp only [Safety.traceSafeList_cons,singleton]
  simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,safeBlockPtr2]
  simp [Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    ctx.it,ctx.ic,ctx.bh,ctx.ii,ctx.ij,BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,
    TileShape.indexToList]
  intro i j row column
  convert lt_of_lt_of_le (FPReadBounds.tensor_bound capacity.head row column) capacity.output using 1 <;> ring

theorem inter_range (M : Algebra α) (D : FPOutput.DotPrimitive α) (c : ProducerGeometry.Shape) (scale : ℝ) (bounds : RegionName → Nat)
    (initial current : State α) (ctx : Context c initial current) (a : Index → α) (ha : HasAcc a current)
    (capacity : InterCapacity c initial bounds) (hen : enabled c initial) :
    Safety.rangeSafe (engine (FPOutput.withDot M D)) bounds "i_k" 0 (Geometry.ceilDiv c.K c.BK) 1 (parts c scale).body current := by
  apply Safety.rangeSafe_of_invariant (engine (FPOutput.withDot M D)) bounds "i_k" (Geometry.ceilDiv c.K c.BK) 1 (parts c scale).body
    (fun _ t => ∃ a,Context c initial t ∧ HasAcc a t)
  · intro f active t h
    obtain ⟨a,ctx,ha⟩ := h
    exact inter_body M D c scale bounds initial t f ctx capacity hen
  · intro f active t h u ran
    obtain ⟨a,ctx,ha⟩ := h
    have equal := Option.some.inj (ran.symm.trans (bodyExecution M D c scale initial t f a ctx ha).property)
    subst u
    exact ⟨update M D c scale initial f a,bodyContext M D c scale initial t f a ctx ha⟩
  · exact ⟨a,ctx,ha⟩

theorem inter_source (M : Algebra α) (D : FPOutput.DotPrimitive α)
    (c : ProducerGeometry.Shape) (scale : ℝ) (bounds : RegionName → Nat)
    (s : State α) (capacity : InterCapacity c s bounds) :
    TraceSafe (engine (FPOutput.withDot M D)) bounds (kernel c scale) s := by
  let start := entryExecution M D c scale s
  have hs : run (engine (FPOutput.withDot M D)) (parts c scale).before (prologue M D c scale s).val=some start.val := by
    simpa only [run_append,(prologue M D c scale s).property,Option.bind_some] using start.property
  change Safety.traceSafeList (engine (FPOutput.withDot M D)) bounds (kernel c scale).surfaceBody s
  rw [source_parts,Safety.traceSafeList_append]
  refine ⟨inter_prologue M D c scale bounds s,?_⟩
  intro t ran
  have same := Option.some.inj (ran.symm.trans (prologue M D c scale s).property)
  subst t
  rw [singleton,Safety.stepSafe_unfold]
  constructor
  · simp [outer,kernel,SourceBridge.interNew,TypedInter.original,TypedInter.source,
      ProducerGeometry.layout,ProducerGeometry.regions,ComputeKernel.surfaceBody,
      Source.conditional?,Source.prefixBefore,Op.tritonDiv,Op.tritonMod,Safety.safeOp_unfold]
  · rw [outer_condition]
    by_cases time : rowStart c s<c.T
    · simp only [time,decide_true,ite_true]
      rw [singleton,Safety.stepSafe_unfold]
      constructor
      · simp [FPInter.inner,outer,kernel,SourceBridge.interNew,TypedInter.original,TypedInter.source,
          ProducerGeometry.layout,ProducerGeometry.regions,ComputeKernel.surfaceBody,
          Source.conditional?,Source.prefixBefore,Op.tritonDiv,Op.tritonMod,Safety.safeOp_unfold]
      · rw [inner_condition]
        by_cases lower : s.pids 1%c.NC<s.pids 1/c.NC
        · simp only [lower,decide_true,ite_true]
          rw [List.append_assoc,Safety.traceSafeList_append]
          refine ⟨inter_entry M D c scale bounds _,?_⟩
          intro t ran
          have same := Option.some.inj (ran.symm.trans hs)
          subst t
          obtain ⟨mid,executed,ctx,acc⟩ := loopExecution M D c scale s start.val
            (entryContext M D c scale s).1 (entryContext M D c scale s).2
          have h0 : (parts c scale).start = .constNat 0 := rfl
          have hn : (parts c scale).stop = .div .nat .nil
            (.sub .nat .nil (.add .nat .nil (.constNat c.K) (.constNat c.BK)) (.constNat 1)) (.constNat c.BK) := rfl
          have h1 : (parts c scale).stride = .constNat 1 := rfl
          have hl : step (engine (FPOutput.withDot M D))
              (.forRangeDyn "i_k" (parts c scale).start (parts c scale).stop
                (parts c scale).stride (parts c scale).body) start.val=some mid := by
            simpa only [step,h0,hn,h1,evalOp_unfold,Option.bind,Option.bind_some,numeric,bop] using executed
          simp only [List.singleton_append]
          apply Safety.traceSafeList_cons_intro
          · simp only [Safety.stepSafe_unfold,h0,hn,h1,Safety.safeOp_unfold,evalOp_unfold,
              Option.bind,Option.bind_some,numeric,bop,true_and]
            exact inter_range M D c scale bounds s start.val (entryContext M D c scale s).1 _
              (entryContext M D c scale s).2 capacity ⟨time,lower⟩
          · intro t ran
            have same := Option.some.inj (ran.symm.trans hl)
            subst t
            exact inter_finish M D c scale bounds s mid ctx capacity
        · simp [lower]
    · simp [time]

#axiomsClean inter_finish
#axiomsClean inter_source
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeSafety
