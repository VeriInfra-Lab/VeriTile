import VeriTile.Triton.Concurrent.EventProgramIO
import VeriTile.Triton.Concurrent.EventLocality
import VeriTile.Triton.Concurrent.EventSafety

/-! Boundary transport for finite concurrent IO. A canonical finite-input
proof may be reused only through bounded heap agreement and a proved safe
trace. Actual execution still starts from the arbitrary supplied heaps. -/
namespace VeriTile.Triton.Concurrent

theorem Window.InBounds.mono (window : Window) (small large : Nat → Nat)
    (capacity : ∀ region,small region≤large region) (safe : window.InBounds small) :
    window.InBounds large :=
  fun lane active=>InWindow.mono (capacity window.region) (safe lane active)

namespace Locality

/-- Source-owned registers and CAS return slots are initialized by launch;
their arbitrary incoming values need not agree. All three heaps remain intact. -/
theorem initialValues_agree (b : Backend V) (bounds : Bounds) (s t : State V)
    (inputs : HeapAgree bounds.input s.input t.input)
    (mem : HeapAgree bounds.mem s.mem t.mem)
    (words : HeapAgree bounds.words s.words t.words) :
    Agree bounds (Program.initialValues b s) (Program.initialValues b t) :=
  ⟨inputs,mem,words,rfl,rfl⟩

end Locality

namespace Program

/-- Reuse a constructive completed controller execution on bounded-equivalent
heaps. The same safe event trace is emitted and completion is transported. -/
theorem controller_runs_transfer (b : Backend V) (bounds : Bounds) (control : Controller)
    (s t final : State V) (events : List Event)
    (inputs : Locality.HeapAgree bounds.input s.input t.input)
    (mem : Locality.HeapAgree bounds.mem s.mem t.mem)
    (words : Locality.HeapAgree bounds.words s.words t.words)
    (ran : Program.Runs b (.controller control) s events final)
    (safe : TapeSafe bounds events) :
    ∃ other,Program.Runs b (.controller control) t events other ∧ Locality.Agree bounds final other := by
  obtain ⟨schedule,completed,completedRun,done,rfl⟩ := ran
  have initial := Locality.initial_agree bounds control _ _
    (Locality.initialValues_agree b bounds s t inputs mem words)
  obtain ⟨other,otherRun,agree⟩ := Locality.run_transfer b bounds control schedule
    _ _ events completed initial completedRun safe
  exact ⟨other.values,⟨schedule,other,otherRun,
    (Locality.done_iff bounds control completed other agree).mp done,rfl⟩,agree.values⟩

/-- Tapes use their actual supplied register state. Controllers additionally
apply their explicit fixed launch initialization before the first event. -/
theorem runs_transfer (b : Backend V) (bounds : Bounds) (program : Program)
    (s t final : State V) (events : List Event) (agree : Locality.Agree bounds s t)
    (ran : program.Runs b s events final) (safe : TapeSafe bounds events) :
    ∃ other,program.Runs b t events other ∧ Locality.Agree bounds final other := by
  cases program with
  | tape instructions =>
    obtain ⟨rfl,rfl⟩ := ran
    exact ⟨run b events t,⟨rfl,rfl⟩,Locality.run_agree b bounds events s t agree safe⟩
  | controller control =>
    exact controller_runs_transfer b bounds control s t final events
      agree.input agree.mem agree.words ran safe

end Program

namespace EventProgramIO
variable {Source Input Output V : Type}

theorem InBounds.mono (io : EventProgramIO Source Input Output V) (small large : Bounds)
    (capacity : small.LE large) (safe : io.InBounds small) : io.InBounds large := by
  refine ⟨?_,?_,?_,?_,?_,?_⟩
  · intro port member
    have old := safe.1 port member
    cases space : port.space with
    | input =>
      simp only [space] at old ⊢
      exact Window.InBounds.mono port.window small.input large.input capacity.input old
    | mem =>
      simp only [space] at old ⊢
      exact Window.InBounds.mono port.window small.mem large.mem capacity.mem old
  · intro port member
    exact Window.InBounds.mono port small.words large.words capacity.words (safe.2.1 port member)
  · intro port member
    exact Window.InBounds.mono port small.mem large.mem capacity.mem (safe.2.2.1 port member)
  · intro port member
    exact Window.InBounds.mono port small.words large.words capacity.words (safe.2.2.2.1 port member)
  · intro port member
    exact Window.InBounds.mono port small.mem large.mem capacity.mem (safe.2.2.2.2.1 port member)
  · intro port member
    exact Window.InBounds.mono port small.words large.words capacity.words (safe.2.2.2.2.2 port member)

theorem InputBinding.transfer (io : EventProgramIO Source Input Output V)
    (bounds : Bounds) (values : io.Inputs) (s t : State V)
    (bounded : io.InBounds bounds) (agree : Locality.Agree bounds s t)
    (binding : io.InputBinding values s) : io.InputBinding values t := by
  refine ⟨?_,?_⟩
  · intro port lane active
    have old := binding.1 port lane active
    have access := bounded.1 io.scalarInputs[port] (List.get_mem _ port) lane active
    cases space : io.scalarInputs[port].space with
    | input =>
      simp only [space] at old access ⊢
      rw [←agree.input _ _ access]
      exact old
    | mem =>
      simp only [space] at old access ⊢
      rw [←agree.mem _ _ access]
      exact old
  · intro port lane active
    have access := bounded.2.1 io.wordInputs[port] (List.get_mem _ port) lane active
    rw [←agree.words _ _ access]
    exact binding.2 port lane active

theorem OutputBinding.transfer (io : EventProgramIO Source Input Output V)
    (bounds : Bounds) (values : io.Outputs) (s t : State V)
    (bounded : io.InBounds bounds) (agree : Locality.Agree bounds s t)
    (binding : io.OutputBinding values s) : io.OutputBinding values t := by
  refine ⟨?_,?_⟩
  · intro port lane active
    have access := bounded.2.2.1 io.scalarOutputs[port] (List.get_mem _ port) lane active
    rw [←agree.mem _ _ access]
    exact binding.1 port lane active
  · intro port lane active
    have access := bounded.2.2.2.1 io.wordOutputs[port] (List.get_mem _ port) lane active
    rw [←agree.words _ _ access]
    exact binding.2 port lane active

end EventProgramIO

#axiomsClean Program.controller_runs_transfer
#axiomsClean Program.runs_transfer
#axiomsClean EventProgramIO.InBounds.mono
#axiomsClean EventProgramIO.InputBinding.transfer
#axiomsClean EventProgramIO.OutputBinding.transfer
end VeriTile.Triton.Concurrent
