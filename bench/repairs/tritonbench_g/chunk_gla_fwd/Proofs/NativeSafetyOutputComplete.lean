import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeSafetyOutput

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite FPOutput
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 4000000
set_option maxRecDepth 30000

theorem output_range (M : Algebra α) (D : DotPrimitive α) (c : OutputGrid.Shape) (scale : ℝ)
    (bounds : RegionName → Nat) (initial current : State α) (ctx : Context c initial current)
    (a : Index c → α) (ha : HasAcc c a current) (capacity : OutputCapacity c initial bounds) :
    Safety.rangeSafe (engine (withDot M D)) bounds "i_k" 0 (Geometry.ceilDiv c.K c.BK) 1
      (parts c scale).body current := by
  apply Safety.rangeSafe_of_invariant (engine (withDot M D)) bounds "i_k"
    (Geometry.ceilDiv c.K c.BK) 1 (parts c scale).body
    (fun _ t => ∃ a,Context c initial t ∧ HasAcc c a t)
  · intro f active t h
    obtain ⟨a,ctx,ha⟩ := h
    exact output_body M D c scale bounds initial t f ctx capacity
  · intro f active t h u ran
    obtain ⟨a,ctx,ha⟩ := h
    have same := Option.some.inj (ran.symm.trans (bodyExecution M D c scale initial t f a ctx ha).property)
    subst u
    exact ⟨update M D c scale initial f a,bodyContext M D c scale initial t f a ctx ha⟩
  · exact ⟨a,ctx,ha⟩

theorem output_source (M : Algebra α) (D : DotPrimitive α) (c : OutputGrid.Shape) (scale : ℝ)
    (bounds : RegionName → Nat) (s : State α) (capacity : OutputCapacity c s bounds) :
    TraceSafe (engine (withDot M D)) bounds (kernel c scale) s := by
  let start := entryExecution M D c scale s
  obtain ⟨mid,executed,ctx,acc⟩ := loopExecution M D c scale s start.val
    (entryContext M D c scale s).1 (entryContext M D c scale s).2
  change Safety.traceSafeList (engine (withDot M D)) bounds (kernel c scale).surfaceBody s
  rw [source_parts,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨output_entry M D c scale bounds s,?_⟩
  intro t ran
  have same := Option.some.inj (ran.symm.trans start.property)
  subst t
  have h0 : (parts c scale).start = .constNat 0 := rfl
  have hn : (parts c scale).stop = .div .nat .nil
    (.sub .nat .nil (.add .nat .nil (.constNat c.K) (.constNat c.BK)) (.constNat 1)) (.constNat c.BK) := rfl
  have h1 : (parts c scale).stride = .constNat 1 := rfl
  have hl : step (engine (withDot M D)) (.forRangeDyn "i_k" (parts c scale).start
      (parts c scale).stop (parts c scale).stride (parts c scale).body) start.val=some mid := by
    simpa only [step,h0,hn,h1,evalOp_unfold,Option.bind,Option.bind_some,numeric,bop] using executed
  simp only [List.singleton_append]
  apply Safety.traceSafeList_cons_intro
  · simp only [Safety.stepSafe_unfold,h0,hn,h1,Safety.safeOp_unfold,evalOp_unfold,
      Option.bind,Option.bind_some,numeric,bop,true_and]
    exact output_range M D c scale bounds s start.val (entryContext M D c scale s).1 _
      (entryContext M D c scale s).2 capacity
  · intro t ran
    have same := Option.some.inj (ran.symm.trans hl)
    subst t
    exact output_finish M D c scale bounds s mid ctx _ acc capacity

#axiomsClean output_source
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeSafety
