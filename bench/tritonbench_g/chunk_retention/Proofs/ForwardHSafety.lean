import bench.tritonbench_g.chunk_retention.Proofs.ForwardHAccessEntry
import bench.tritonbench_g.chunk_retention.Proofs.ForwardHAccessSuffix
import bench.tritonbench_g.chunk_retention.Proofs.ForwardHAccessPrelude
import bench.tritonbench_g.chunk_retention.Proofs.ForwardHAccessFinish
import bench.tritonbench_g.chunk_retention.Proofs.AccessSafetyLoops

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardHSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

/-- Actual forward-history source safety, including seed loads, history
stores before every carry update, tail-block loads and the optional final store. -/
theorem source_safe (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (seed final : Bool) (initial : State α) (bounds : RegionName → Nat)
    (h0 : seed=Bool.true → ∀ i,active c initial i → stateAddress c initial i<bounds "h0")
    (hk : ∀ e t,e<c.K → t<c.T → initial.pids 2*(c.T*c.K)+e+t*c.K<bounds "k")
    (hv : ∀ t j,t<c.T → j<c.V → initial.pids 2*(c.T*c.V)+t*c.V+j<bounds "v")
    (hh : ∀ n,n<c.chunks → ∀ i,active c initial i → historyAddress c initial n i<bounds "h")
    (ht : final=Bool.true → ∀ i,active c initial i → stateAddress c initial i<bounds "ht") :
    TraceSafe (engine (withDot M D)) bounds (kernel c seed final) initial := by
  let I := fun n current => Context M c initial current n ∧ ∃ carry,HasCarry c carry current
  have loopSafe (current : State α) (ctx : Context M c initial current 0)
      (ha : HasCarry c (initValue M c seed initial) current) :
      Safety.rangeSafe (engine (withDot M D)) bounds "i_t" 0 c.chunks 1
        (parts c seed final).body current := by
    apply AccessSafety.range_safe _ bounds "i_t" c.chunks _ I _ 0 current ⟨ctx,_,ha⟩
    intro n s hn h
    obtain ⟨carry,hcarry⟩ := h.2
    let prep := preludeExecution M D c seed final initial s n carry h.1 hcarry
    let hp := preludePrepared M D c seed final initial s n carry h.1 hcarry
    let finish := suffixExecution M D c seed final initial prep.val n carry hn hp
    constructor
    · rw [body_parts,Safety.traceSafeList_append]
      refine ⟨prelude_safe M D c seed final initial s n carry h.1 hcarry bounds (hh n hn),?_⟩
      intro t ran
      rw [prep.property] at ran
      cases ran
      exact suffix_safe M D c seed final initial prep.val n carry hn hp bounds hk hv
    · refine ⟨finish.val,?_,suffixContext M D c seed final initial prep.val n carry hn hp,
        _,suffixCarry M D c seed final initial prep.val n carry hn hp⟩
      rw [body_parts,run_append]
      conv_lhs => rw [prep.property]
      exact finish.property
  unfold TraceSafe
  rw [source_parts,Safety.traceSafeList_append,Safety.traceSafeList_append]
  let start := entryExecution M D c seed final initial
  have hc := entryContext M D c seed final initial
  have hmem : start.val.mem=initial.mem := by cases seed <;> simp [start,entryExecution]
  obtain ⟨mid,hm,⟨n,ctx⟩,ha,hmm⟩ := loopExecution M D c seed final initial start.val hc.1 hc.2 hmem
  have h0' : (parts c seed final).start=.constNat 0 := rfl
  have hn' : (parts c seed final).stop=.constNat c.chunks := rfl
  have h1' : (parts c seed final).stride=.constNat 1 := rfl
  refine ⟨⟨entry_safe M D c seed final initial bounds h0,?_⟩,?_⟩
  · intro t ran
    rw [start.property] at ran
    cases ran
    apply Safety.traceSafeList_cons_intro
    · simpa [Safety.stepSafe_unfold,Safety.safeOp_unfold,h0',hn',h1',evalOp_unfold]
        using loopSafe start.val hc.1 hc.2
    · intro _ _; exact Safety.traceSafeList_nil _ _ _
  · intro t ran
    rw [run_append,start.property] at ran
    simp only [Option.bind_some,run,step,h0',hn',h1',evalOp_unfold] at ran
    change (range (engine (withDot M D)) "i_t" 0 c.chunks 1 (parts c seed final).body start.val).bind
      (fun s => some s)=some t at ran
    rw [hm] at ran
    simp only [Option.bind_some] at ran
    cases ran
    exact finish_safe M D c seed final initial mid n _ ctx ha bounds ht

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardHSource
