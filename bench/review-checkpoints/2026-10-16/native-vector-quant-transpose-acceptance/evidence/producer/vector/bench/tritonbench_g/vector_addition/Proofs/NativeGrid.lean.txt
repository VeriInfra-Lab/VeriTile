import bench.tritonbench_g.vector_addition.Proofs.NativeProgram

namespace VeriTile.Bench.TritonBenchG.VectorAddition.Native
open VeriTile.Triton DTypeScope TypedHost Finite
noncomputable section
open Classical
set_option maxHeartbeats 10000000
set_option maxRecDepth 24000

def gridPlan {M : Type} (r : Route) (ops : Memory M (Carrier r)) (b : Numerics)
    (p : Layout) (n : Nat) (snapshot : M) : List Nat → List Nat → Except Issue (List (Write (Carrier r))×List Nat)
  | [],seen => .ok ([],seen)
  | pid::rest,seen => do
      let (head,owners) ← programPlan r ops b p n snapshot pid seen
      let (tail,finalOwners) ← gridPlan r ops b p n snapshot rest owners
      pure (head++tail,finalOwners)

def executeGrid {M : Type} (r : Route) (ops : Memory M (Carrier r)) (b : Numerics)
    (p : Layout) (n : Nat) : List Nat → State M → Except Issue (State M)
  | [],state => .ok state
  | pid::rest,state => do
      let next ← executeProgram r ops b p n pid state
      executeGrid r ops b p n rest next

theorem grid_correspondence {M : Type} (r : Route) (ops : Memory M (Carrier r)) (b : Numerics)
    (p : Layout) (n : Nat) (snapshot : M) (fresh : p.Fresh) (pids : List Nat)
    (state : State M) (same : OutsideOutput ops p state.memory snapshot) :
    executeGrid r ops b p n pids state=(gridPlan r ops b p n snapshot pids state.owners).map
      (fun plan => ⟨applyWrites ops r.signature.1 plan.1 state.memory,plan.2⟩) := by
  induction pids generalizing state with
  | nil => rfl
  | cons pid rest ih =>
      simp only [executeGrid,gridPlan,program_correspondence]
      rw [program_plan_frame r ops b p n state.memory snapshot same fresh]
      cases hh : programPlan r ops b p n snapshot pid state.owners with
      | error issue => rfl
      | ok head =>
          rcases head with ⟨head,owners⟩
          simp only [Except.map,bind_ok]
          have hb := program_write_bounds r ops b p n snapshot pid state.owners head owners hh
          have hs : OutsideOutput ops p (applyWrites ops r.signature.1 head state.memory) snapshot :=
            outside_trans ops p _ _ _ (writes_outside ops p r.signature.1 head state.memory hb) same
          rw [ih ⟨applyWrites ops r.signature.1 head state.memory,owners⟩ hs]
          cases gridPlan r ops b p n snapshot rest owners with
          | error issue => rfl
          | ok tail =>
              rcases tail with ⟨tail,finalOwners⟩
              simp only [Except.map,bind_ok,pure_ok,writes_append]

theorem grid_write_bounds {M : Type} (r : Route) (ops : Memory M (Carrier r)) (b : Numerics)
    (p : Layout) (n : Nat) (snapshot : M) (pids : List Nat) (seen : List Nat)
    (writes : List (Write (Carrier r))) (owners : List Nat)
    (h : gridPlan r ops b p n snapshot pids seen=.ok (writes,owners)) :
    ∀ w∈writes, p.output.Inside w.address (bytes r.signature.1) := by
  induction pids generalizing seen writes owners with
  | nil => cases h; simp
  | cons pid rest ih =>
      simp only [gridPlan] at h
      cases hh : programPlan r ops b p n snapshot pid seen with
      | error issue => simp [hh] at h
      | ok head =>
          rcases head with ⟨head,nextOwners⟩
          cases ht : gridPlan r ops b p n snapshot rest nextOwners with
          | error issue => simp [hh,ht] at h
          | ok tail =>
              rcases tail with ⟨tail,finalOwners⟩
              simp [hh,ht] at h
              rcases h with ⟨rfl,rfl⟩
              intro w hw
              rcases List.mem_append.1 hw with hw|hw
              · exact program_write_bounds r ops b p n snapshot pid seen head nextOwners hh w hw
              · exact ih _ _ _ ht w hw

def launchPlan {M : Type} (r : Route) (ops : Memory M (Carrier r)) (b : Numerics)
    (p : Layout) (n : Nat) (snapshot : M) : Except Issue (List (Write (Carrier r))) :=
  (gridPlan r ops b p n snapshot (List.range (grid n)) []).map Prod.fst
def executeLaunch {M : Type} (r : Route) (ops : Memory M (Carrier r)) (b : Numerics)
    (p : Layout) (n : Nat) (memory : M) : Except Issue M :=
  (executeGrid r ops b p n (List.range (grid n)) ⟨memory,[]⟩).map State.memory

theorem launch_correspondence {M : Type} (r : Route) (ops : Memory M (Carrier r)) (b : Numerics)
    (p : Layout) (n : Nat) (memory : M) (fresh : p.Fresh) :
    executeLaunch r ops b p n memory=(launchPlan r ops b p n memory).map
      (fun writes => applyWrites ops r.signature.1 writes memory) := by
  unfold executeLaunch launchPlan
  rw [grid_correspondence r ops b p n memory fresh _ _ (outside_refl ops p memory)]
  cases gridPlan r ops b p n memory (List.range (grid n)) [] <;> rfl

theorem launch_write_bounds {M : Type} (r : Route) (ops : Memory M (Carrier r)) (b : Numerics)
    (p : Layout) (n : Nat) (memory : M) (writes : List (Write (Carrier r)))
    (h : launchPlan r ops b p n memory=.ok writes) :
    ∀ w∈writes, p.output.Inside w.address (bytes r.signature.1) := by
  unfold launchPlan at h
  cases hg : gridPlan r ops b p n memory (List.range (grid n)) [] with
  | error issue => simp [hg,Except.map] at h
  | ok plan =>
      simp only [hg,Except.map,Except.ok.injEq] at h
      subst writes
      exact grid_write_bounds r ops b p n memory _ _ _ _ hg

/-- No safety/definedness predicate restricts the master launch. Every finite
request either yields a recorded observation or an independent write formula. -/
theorem launch_all_outcomes {M : Type} (r : Route) (ops : Memory M (Carrier r)) (b : Numerics)
    (p : Layout) (n : Nat) (memory : M) (fresh : p.Fresh) :
    (∃ issue, launchPlan r ops b p n memory=.error issue ∧ executeLaunch r ops b p n memory=.error issue) ∨
    (∃ writes, launchPlan r ops b p n memory=.ok writes ∧
      executeLaunch r ops b p n memory=.ok (applyWrites ops r.signature.1 writes memory) ∧
      OutsideOutput ops p (applyWrites ops r.signature.1 writes memory) memory ∧
      (∀ address, (∀ w∈writes, ¬w.Touches r.signature.1 address) →
        ops.byte (applyWrites ops r.signature.1 writes memory) address=ops.byte memory address)) := by
  have hc := launch_correspondence r ops b p n memory fresh
  cases ht : launchPlan r ops b p n memory with
  | error issue => exact Or.inl ⟨issue,rfl,by simpa [ht,Except.map] using hc⟩
  | ok writes =>
      exact Or.inr ⟨writes,rfl,by simpa [ht,Except.map] using hc,
        writes_outside ops p r.signature.1 writes memory (launch_write_bounds r ops b p n memory writes ht),
        fun address outside => writes_byte_frame ops r.signature.1 writes memory address outside⟩

theorem no_access_grid {M : Type} (r : Route) (ops : Memory M (Carrier r)) (b : Numerics)
    (p : Layout) (memory : M) (pids : List Nat) (seen : List Nat) :
    gridPlan r ops b p (2^31) memory pids seen=.ok ([],seen) := by
  induction pids generalizing seen with
  | nil => rfl
  | cons pid rest ih => simp only [gridPlan,no_access_program,bind_ok,ih,pure_ok,List.nil_append]
theorem no_access_returned_memory {M : Type} (r : Route) (ops : Memory M (Carrier r)) (b : Numerics)
    (p : Layout) (memory : M) : executeLaunch r ops b p (2^31) memory=.ok memory := by
  have programs : ∀ pid (state : State M), executeProgram r ops b p (2^31) pid state=.ok state := by
    intro pid state
    rw [program_correspondence,no_access_program]
    rfl
  have grids : ∀ pids (state : State M), executeGrid r ops b p (2^31) pids state=.ok state := by
    intro pids
    induction pids with
    | nil => intro state; rfl
    | cons pid rest ih => intro state; simp only [executeGrid,programs,bind_ok,ih]
  rw [executeLaunch,grids]
  rfl

#specNonCircular launchPlan avoiding [executeLaunch,executeGrid,executeProgram,sourceWrites,applyWrites]
#axiomsClean grid_correspondence
#axiomsClean launch_correspondence
#axiomsClean launch_write_bounds
#axiomsClean launch_all_outcomes
#axiomsClean no_access_grid
#axiomsClean no_access_returned_memory
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAddition.Native
