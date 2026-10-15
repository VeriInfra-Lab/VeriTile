import bench.tritonbench_g.layer_norm_triton.Proofs.ConcurrentValues

namespace VeriTile.Bench.TritonBenchG.LayerNormLock

/-- Direct event frame; inactive-lane arithmetic is allowed and does not turn
into a memory write. -/
def Untouched (e : Event) (region : Region) (address : Int) : Prop :=
  match e with
  | .dx _ _ a => region ≠ .dx ∨ address ≠ a
  | .storeDW _ _ a => region ≠ .dw ∨ address ≠ a
  | .storeDB _ _ a => region ≠ .db ∨ address ≠ a
  | _ => True

def Shared.read (s : Shared V) (region : Region) (address : Int) (fallback : V) : V :=
  match region with
  | .dx => s.dx address
  | .dw => s.dw address
  | .db => s.db address
  | _ => fallback

theorem execute_frame (b : Backend V) (c : Config) (rows : Nat → RowValues V)
    (undef : Undef V) (e : Event) (s : Values V) (region : Region) (address : Int)
    (fallback : V) (h : Untouched e region address) :
    (execute b c rows undef e s).mem.read region address fallback = s.mem.read region address fallback := by
  cases e <;> cases region <;> simp_all [Untouched, execute, Shared.read, put]

theorem run_frame (b : Backend V) (c : Config) (rows : Nat → RowValues V)
    (undef : Undef V) (schedule : List Nat) (s : State V) (region : Region) (address : Int)
    (fallback : V) (h : ∀ e ∈ (run b c rows undef schedule s).1, Untouched e region address) :
    (run b c rows undef schedule s).2.values.mem.read region address fallback =
      s.values.mem.read region address fallback := by
  induction schedule generalizing s with
  | nil => rfl
  | cons p ps ih =>
    have ht : ∀ e ∈ (run b c rows undef ps (step b c rows undef p s).2).1, Untouched e region address :=
      fun e he => h e (List.mem_cons_of_mem _ he)
    rw [show (run b c rows undef (p :: ps) s).2 = (run b c rows undef ps (step b c rows undef p s).2).2 from rfl,
      ih _ ht]
    exact execute_frame b c rows undef (controlStep c p s.control).1 s.values region address fallback
      (h _ (List.mem_cons_self ..))

theorem failed_poll_control (c : Config) (p : Nat) (s : Control)
    (hp : p < c.grid) (hpc : s.pc p = .poll) (held : s.words (lockAddr c p) = 1) :
    controlStep c p s = (.poll (lockAddr c p), s) := by
  have hwords : put s.words (lockAddr c p) 1 = s.words := by
    funext a; by_cases ha : a = lockAddr c p <;> simp [put, ha, held]
  have hpcs : put s.pc p Phase.poll = s.pc := by
    funext q; by_cases hq : q = p <;> simp [put, hq, hpc]
  simp only [controlStep, decode, Nat.not_le_of_lt hp, if_false, hpc, held]
  simp only [beq_self_eq_true, if_true, integerEffect, cas01]
  simp only [held]
  norm_num
  rw [hwords, hpcs]

theorem failed_poll_step (b : Backend V) (c : Config) (rows : Nat → RowValues V)
    (undef : Undef V) (p : Nat) (s : State V)
    (hp : p < c.grid) (hpc : s.control.pc p = .poll) (held : s.control.words (lockAddr c p) = 1) :
    step b c rows undef p s = (.poll (lockAddr c p), s) := by
  simp only [step, failed_poll_control c p s.control hp hpc held, execute]

theorem failed_poll_forever (b : Backend V) (c : Config) (rows : Nat → RowValues V)
    (undef : Undef V) (p retries : Nat) (s : State V)
    (hp : p < c.grid) (hpc : s.control.pc p = .poll) (held : s.control.words (lockAddr c p) = 1) :
    run b c rows undef (List.replicate retries p) s = (List.replicate retries (.poll (lockAddr c p)), s) := by
  induction retries with
  | zero => rfl
  | succ n ih => simp only [List.replicate_succ, run, failed_poll_step b c rows undef p s hp hpc held, ih]

#axiomsClean execute_frame
#axiomsClean run_frame
#axiomsClean failed_poll_forever
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LayerNormLock
