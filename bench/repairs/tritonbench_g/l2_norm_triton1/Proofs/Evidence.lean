import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.PublicFacts

namespace VeriTile.Bench.Repairs.L2NormTriton1.Host
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Physical
noncomputable section
open Classical
set_option maxHeartbeats 6000000

/-- Execution certificates use the actual setup heap and schedule, and every
prefix of that schedule. Empty rows perform no setup or kernel launch. -/
def LaunchEvidence {M α β γ : Type} [Inhabited α] [Inhabited γ]
    (ops : Ops M α β) (a : Source.Adapter α γ) (rt : Runtime M)
    (q : Prepared) (eps : γ) (before after : World M) : Prop :=
  q.layout.rows=0 ∨ ∃ schedule allRows ready,
    rt.setup q.layout before=.ok (.complete schedule allRows) ready ∧
    after={ready with memory:=Physical.run ops (a.compute q.layout eps) q.layout schedule ready.memory} ∧
    Source.Certificates ops a q.layout eps schedule ready.memory

theorem launch_evidence {M α β γ : Type} [Inhabited α] [Inhabited γ]
    (ops : Ops M α β) (a : Source.Adapter α γ) (rt : Runtime M)
    (q : Prepared) (valid : Valid q.layout) (eps : γ) (before after : World M)
    (success : launch ops a rt q eps before=.ok () after) :
    LaunchEvidence ops a rt q eps before after := by
  by_cases empty : q.layout.rows=0
  · exact Or.inl empty
  · right
    cases hs : rt.setup q.layout before with
    | error reason ready => simp [launch,empty,hs,Result.bind] at success
    | ok plan ready =>
      cases plan with
      | interrupted completed storesDone bounded partialBound reason =>
        simp [launch,empty,hs,Result.bind] at success
      | complete schedule allRows =>
        simp only [launch,empty,if_false,hs,Result.bind,Result.ok.injEq] at success
        obtain ⟨_,rfl⟩ := success
        exact ⟨schedule,allRows,ready,rfl,rfl,Source.certificates ops a q.layout valid eps schedule
          (fun row hr => List.mem_range.mp (allRows.mem_iff.mp hr)) ready.memory⟩

def ExecutionEvidence {M α β γ : Type} [Inhabited α] [Inhabited γ]
    (ops : Ops M α β) (neg : Negation α) (a : Source.Adapter α γ) (rt : Runtime M)
    (call : Call γ) (entry : World M) (out : View) (after : World M) : Prop :=
  ∃ token entered q ready done,
    rt.enter call.device entry=.ok token entered ∧ prepare rt call entered=.ok q ready ∧
    launch ops a rt q call.eps ready=.ok () done ∧ rt.leave token done=.ok () after ∧
    rt.reshape q.output call.input.shape=some out ∧
    PreparedFacts ops neg call entered q ready ∧ LaunchEvidence ops a rt q call.eps ready done

theorem execution_evidence {M α β γ : Type} [Inhabited α] [Inhabited γ]
    (ops : Ops M α β) (neg : Negation α) (a : Source.Adapter α γ) (rt : Runtime M)
    (lib : Library ops neg rt) (call : Call γ) (entry : World M) (domain : Domain call entry)
    (out : View) (after : World M) (trace : SuccessTrace ops a rt call entry out after) :
    ExecutionEvidence ops neg a rt call entry out after := by
  obtain ⟨_,token,entered,q,ready,done,he,hp,hk,hl,hr⟩ := trace
  have keep : Keep ops entry entered := by simpa [he,Result.state] using lib.enter call.device entry
  have facts := prepare_sound ops neg rt lib call entered
    ⟨domain.backed,keep.live _ domain.live,domain.device,domain.direct⟩ q ready hp
  exact ⟨token,entered,q,ready,done,he,hp,hk,hl,hr,facts,
    launch_evidence ops a rt q facts.valid call.eps ready done hk⟩

/-- Address certificates follow from the actual same-call allocations. The
padded signed64 bound and active byte representability are separate from any
runtime grid-size or native program-ID compatibility assumption. -/
theorem prepared_addresses {M α β γ : Type} (ops : Ops M α β) (neg : Negation α)
    (call : Call γ) (entry : World M) (q : Prepared) (ready : World M)
    (facts : PreparedFacts ops neg call entry q ready) :
    (∀ row col,row<q.layout.rows → col<q.layout.cols →
      row*q.layout.cols+col<2^62 ∧ (q.layout.xkey row col).address+4≤2^64 ∧
        (q.layout.ykey row col).address+4≤2^64) ∧
    (∀ row lane,row<q.layout.rows → lane<q.layout.block → row*q.layout.cols+lane<2^63) ∧
    q.block=2^Nat.clog 2 q.snapshot.cols ∧ Nat.clog 2 q.snapshot.cols≤14 := by
  have accepted : q.snapshot.cols≤16384 := le_trans facts.valid.blockCovers facts.valid.blockLimit
  exact ⟨active_address q.layout facts.valid,padded_index q.layout facts.valid,
    facts.block.trans (block_power _ accepted).1,(block_power _ accepted).2⟩

#axiomsClean execution_evidence
#axiomsClean prepared_addresses
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton1.Host
