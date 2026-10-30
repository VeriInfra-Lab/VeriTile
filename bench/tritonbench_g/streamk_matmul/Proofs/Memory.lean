import bench.tritonbench_g.streamk_matmul.Proofs.Geometry
namespace VeriTile.Bench.Repairs.StreamK.Whole001
namespace Memory
set_option maxHeartbeats 6000000

/-- Local arithmetic produces a value from the two current immutable input
views. Atomic events instead read the actual current output and scratch.
This low-level executor has no final-result or matrix-product callback. -/
@[simp] theorem step_inputs (add : α → α → α) (e : Event α A B C S) (s : State α A B C S E) :
    (step add e s).inputA=s.inputA ∧ (step add e s).inputB=s.inputB ∧
    (step add e s).exterior=s.exterior := by cases e <;> exact ⟨rfl,rfl,rfl⟩

theorem run_inputs (add : α → α → α) (events : List (Event α A B C S)) (s : State α A B C S E) :
    (run add events s).inputA=s.inputA ∧ (run add events s).inputB=s.inputB ∧
    (run add events s).exterior=s.exterior := by
  induction events generalizing s with
  | nil => exact ⟨rfl,rfl,rfl⟩
  | cons e events ih =>
    have h:=ih (step add e s)
    have f:=step_inputs add e s
    exact ⟨h.1.trans f.1,h.2.1.trans f.2.1,h.2.2.trans f.2.2⟩

/-- Explicit kernel completion boundary: the next list consumes the completed
current memory, including actual scratch stores. -/
theorem ordered_launches (add : α → α → α) (first reduce fullWave : List (Event α A B C S))
    (s : State α A B C S E) :
    run add (first++reduce++fullWave) s=run add fullWave (run add reduce (run add first s)) := by
  simp only [run,List.foldl_append]

/-- Each event has a finite address by construction; no out-of-allocation
fallback value or unbounded-memory read is available. -/
def WritesOutput (i : Fin C) : Event α A B C S → Prop
  | .storeOutput j _ | .atomic j _ => j=i
  | .storeScratch _ _ => False

def WritesScratch (i : Fin S) : Event α A B C S → Prop
  | .storeScratch j _ => j=i
  | _ => False

theorem step_output_frame (add : α → α → α) (e : Event α A B C S)
    (s : State α A B C S E) (i : Fin C) (h : ¬WritesOutput i e) :
    (step add e s).output i=s.output i := by
  cases e <;> simp_all [step,WritesOutput,Function.update,ne_comm]

theorem step_scratch_frame (add : α → α → α) (e : Event α A B C S)
    (s : State α A B C S E) (i : Fin S) (h : ¬WritesScratch i e) :
    (step add e s).scratch i=s.scratch i := by
  cases e <;> simp_all [step,WritesScratch,Function.update,ne_comm]

theorem run_output_frame (add : α → α → α) (events : List (Event α A B C S))
    (s : State α A B C S E) (i : Fin C) (h : ∀ e ∈ events, ¬WritesOutput i e) :
    (run add events s).output i=s.output i := by
  induction events generalizing s with
  | nil => rfl
  | cons e es ih =>
    change (run add es (step add e s)).output i=s.output i
    rw [ih (step add e s) (fun e he => h e (List.mem_cons_of_mem _ he))]
    exact step_output_frame add e s i (h e List.mem_cons_self)

theorem run_scratch_frame (add : α → α → α) (events : List (Event α A B C S))
    (s : State α A B C S E) (i : Fin S) (h : ∀ e ∈ events, ¬WritesScratch i e) :
    (run add events s).scratch i=s.scratch i := by
  induction events generalizing s with
  | nil => rfl
  | cons e es ih =>
    change (run add es (step add e s)).scratch i=s.scratch i
    rw [ih (step add e s) (fun e he => h e (List.mem_cons_of_mem _ he))]
    exact step_scratch_frame add e s i (h e List.mem_cons_self)

/-- An atomic reads scratch published by an earlier completed kernel and the
current output of the previous atomic, not the initial output allocation. -/
theorem atomic_current (add : α → α → α) (s : State α A B C S E) (i : Fin C) (j : Fin S) :
    (step add (.atomic i j) s).output i=add (s.output i) (s.scratch j) := by
  simp [step]

def firstOnly : Event α A B C S → Prop
  | .atomic _ _ => False
  | _ => True

def Disjoint (e f : Event α A B C S) : Prop :=
  (∀ i, ¬(WritesOutput i e ∧ WritesOutput i f)) ∧
  (∀ i, ¬(WritesScratch i e ∧ WritesScratch i f))

/-- Disjoint first-wave stores commute for arbitrary payloads. Consequently
first-wave publication is independent of cross-CTA execution order. -/
theorem independent_first_steps (add : α → α → α) (e f : Event α A B C S)
    (s : State α A B C S E) (he : firstOnly e) (hf : firstOnly f) (hd : Disjoint e f) :
    step add e (step add f s)=step add f (step add e s) := by
  cases e with
  | atomic _ _ => simp [firstOnly] at he
  | storeOutput i v =>
    cases f with
    | atomic _ _ => simp [firstOnly] at hf
    | storeScratch j w => rfl
    | storeOutput j w =>
      have hij:i≠j:=by intro hh; subst j; exact hd.1 i ⟨rfl,rfl⟩
      simp only [step]
      congr 1
      exact (Function.update_comm hij _ _ _).symm
  | storeScratch i v =>
    cases f with
    | atomic _ _ => simp [firstOnly] at hf
    | storeOutput j w => rfl
    | storeScratch j w =>
      have hij:i≠j:=by intro hh; subst j; exact hd.2 i ⟨rfl,rfl⟩
      simp only [step]
      congr 1
      exact (Function.update_comm hij _ _ _).symm


/-- Generic finite-store rule, later instantiated by the explicit source
chunk graph. Repeated stores are allowed only when they write the same local
expression; no initial allocation-value premise is needed. -/
theorem scratch_writer_result (add : α → α → α) (events : List (Event α A B C S))
    (s : State α A B C S E) (i : Fin S) (v : (Fin A → α) → (Fin B → α) → α)
    (hw : Event.storeScratch i v ∈ events)
    (unique : ∀ e ∈ events, WritesScratch i e → e=.storeScratch i v) :
    (run add events s).scratch i=v s.inputA s.inputB := by
  induction events generalizing s with
  | nil => simp at hw
  | cons e es ih =>
    simp only [List.mem_cons] at hw
    change (run add es (step add e s)).scratch i=v s.inputA s.inputB
    by_cases ht : Event.storeScratch i v ∈ es
    · have h:=ih (step add e s) ht (fun e he => unique e (List.mem_cons_of_mem _ he))
      have hf:=step_inputs add e s
      rw [hf.1,hf.2.1] at h
      exact h
    · have he:e=.storeScratch i v:=by rcases hw with hw|hw; exact hw.symm; exact False.elim (ht hw)
      subst e
      have frame:∀ e ∈ es, ¬WritesScratch i e:=by
        intro e he hwrite
        have eq:=unique e (List.mem_cons_of_mem _ he) hwrite
        exact ht (eq ▸ he)
      rw [run_scratch_frame add es _ i frame]
      simp [step]

/-- Output initialization rule excludes atomics at this address in the first
wave, exactly as established by the source event constructor. -/
theorem output_writer_result (add : α → α → α) (events : List (Event α A B C S))
    (s : State α A B C S E) (i : Fin C) (v : (Fin A → α) → (Fin B → α) → α)
    (hw : Event.storeOutput i v ∈ events)
    (unique : ∀ e ∈ events, WritesOutput i e → e=.storeOutput i v) :
    (run add events s).output i=v s.inputA s.inputB := by
  induction events generalizing s with
  | nil => simp at hw
  | cons e es ih =>
    simp only [List.mem_cons] at hw
    change (run add es (step add e s)).output i=v s.inputA s.inputB
    by_cases ht : Event.storeOutput i v ∈ es
    · have h:=ih (step add e s) ht (fun e he => unique e (List.mem_cons_of_mem _ he))
      have hf:=step_inputs add e s
      rw [hf.1,hf.2.1] at h
      exact h
    · have he:e=.storeOutput i v:=by rcases hw with hw|hw; exact hw.symm; exact False.elim (ht hw)
      subst e
      have frame:∀ e ∈ es, ¬WritesOutput i e:=by
        intro e he hwrite
        have eq:=unique e (List.mem_cons_of_mem _ he) hwrite
        exact ht (eq ▸ he)
      rw [run_output_frame add es _ i frame]
      simp [step]

#axiomsClean scratch_writer_result
#axiomsClean output_writer_result

#axiomsClean run_inputs
#axiomsClean ordered_launches
#axiomsClean run_output_frame
#axiomsClean run_scratch_frame
#axiomsClean atomic_current
#axiomsClean independent_first_steps
end Memory
end VeriTile.Bench.Repairs.StreamK.Whole001
