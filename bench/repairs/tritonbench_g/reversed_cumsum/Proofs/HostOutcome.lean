import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.Host

namespace VeriTile.Bench.Repairs.ReversedCumsum.HostOutcome
open Host
noncomputable section
set_option maxHeartbeats 3600000

def Good {β : Type} (h : Call) (p : Storage.Placement h)
    (live : Nat → Prop) (s t : Heap β) : Prop :=
  (∀ addr, live addr → t addr=s addr) ∧
  (∀ addr, ¬preparationWrites h p addr → t addr=s addr)

theorem written_preparation (h : Call) (p : Storage.Placement h)
    (addr : Nat)
    (hw : Storage.writes h p (Storage.programs h) addr) : preparationWrites h p addr :=
  ReversedCumsumScalar.Host.output_within_preparation h.flat p _ addr ((Storage.writes_full h p addr).mp hw)

def Result {β : Type} (h : Call) (p : Storage.Placement h)
    (live : Nat → Prop) (caller : Heap β) :
    Except (Failure β) (Allocated β) → Prop
  | .ok a => metadata h p a ∧ Good h p live caller a.heap
  | .error e => e.stage≠.execution ∧ Good h p live caller e.heap

/-- A generic protocol lemma; Real and FP readers instantiate works using the
proved complete grids, never a numerical conformance assumption. -/
theorem outcome {β γ : Type} (api : API β) (h : Call) (p : Storage.Placement h)
    (read : β → γ) (logical : Nat → Nat → Nat → γ)
    (caller : Heap β) (live : Nat → Prop)
    (laws : PreparationLaws api h p read logical caller live)
    (runGrid : Config → Heap β → Option (Heap β))
    (works : ∀ c∈configurations,∀ s,∃ t,runGrid c s=some t ∧
      ∀ addr,¬Storage.writes h p (Storage.programs h) addr → t addr=s addr)
    (admitted : Bool) (tried : List Config) (valid : ∀ c∈tried,c∈configurations)
    (selected : Config) (chosen : selected∈configurations) :
    Result h p live caller (run api runGrid h admitted tried selected caller) := by
  cases prepared : prepare api admitted caller with
  | error e =>
    obtain ⟨stage,fr⟩ := prepare_failure api admitted caller e prepared
    simp only [run,prepared,Result]
    refine ⟨?_,?_⟩
    · rcases stage with eq | eq <;> simp [eq]
    · rw [fr]; exact ⟨fun _ _=>rfl,fun _ _=>rfl⟩
  | ok a =>
    obtain ⟨metadataOk,_,liveOk,frameOk⟩ := prepare_correct api h p read logical caller live laws admitted a prepared
    have prepGood : Good h p live caller a.heap := ⟨liveOk,frameOk⟩
    by_cases empty : h.B*h.H*(h.T*h.S)=0
    · simpa only [run,prepared,if_pos empty,Result] using And.intro metadataOk prepGood
    · have compose : ∀ s t u : Heap β, Good h p live s t → Good h p live t u → Good h p live s u := by
        intro s t u a b
        exact ⟨fun addr hl => (b.1 addr hl).trans (a.1 addr hl),fun addr hn => (b.2 addr hn).trans (a.2 addr hn)⟩
      have launch := launches_outcome api runGrid (Good h p live)
        (fun _ => ⟨fun _ _ => rfl,fun _ _ => rfl⟩) compose 0 (tried++[selected]) (by
          intro c hc s
          have cv : c∈configurations := by
            rcases List.mem_append.mp hc with ht | hs
            · exact valid c ht
            · simp only [List.mem_singleton] at hs
              simpa only [hs] using chosen
          obtain ⟨t,he,frame⟩ := works c cv s
          exact ⟨t,he,fun addr hl => frame addr (laws.fresh addr hl),
            fun addr hn => frame addr (fun hw => hn (written_preparation h p addr hw))⟩) a.heap
      cases le : ReversedCumsumScalar.Host.launches (launchAPI api) runGrid 0 (tried++[selected]) a.heap with
      | ok t =>
        rw [le] at launch
        simp only [run,prepared,if_neg empty,le,Result]
        exact ⟨metadataOk,compose caller a.heap t prepGood launch⟩
      | error e =>
        rw [le] at launch
        simp only [run,prepared,if_neg empty,le,Result]
        exact ⟨by simp [Host.failure,launch.1],compose caller a.heap e.heap prepGood launch.2⟩

#print axioms outcome
end
end VeriTile.Bench.Repairs.ReversedCumsum.HostOutcome
