import bench.tritonbench_g.layer_norm_triton.Proofs.LockControl

namespace VeriTile.Bench.TritonBenchG.LayerNormLock

structure Shared (V : Type) where
  dx : Int → V
  dw : Int → V
  db : Int → V

structure Values (V : Type) where
  mem : Shared V
  dw : Nat → Nat → V
  db : Nat → Nat → V

def initialValues (rows : Nat → RowValues V) (mem : Shared V) : Values V :=
  ⟨mem, fun p => (rows p).dw, fun p => (rows p).db⟩

/-- Pure row programs have already been evaluated against immutable X/DY/W/
Mean/Rstd regions. Their DX stores still occur before each program's CAS.
The actual mutable shared reads, rounded adds and narrow stores remain here. -/
def execute (b : Backend V) (c : Config) (rows : Nat → RowValues V)
    (undef : Undef V) (e : Event) (s : Values V) : Values V :=
  match e with
  | .dx p lane a => { s with mem.dx := put s.mem.dx a ((rows p).dx lane) }
  | .addDW p lane a active =>
    let old := if active then s.mem.dw a else undef .dw a
    { s with dw := put s.dw p (put (s.dw p) lane (b.add (s.dw p lane) (b.cast c.dtype.partialType .fp32 old))) }
  | .addDB p lane a active =>
    let old := if active then s.mem.db a else undef .db a
    { s with db := put s.db p (put (s.db p) lane (b.add (s.db p lane) (b.cast c.dtype.partialType .fp32 old))) }
  | .storeDW p lane a => { s with mem.dw := put s.mem.dw a (b.cast .fp32 c.dtype.partialType (s.dw p lane)) }
  | .storeDB p lane a => { s with mem.db := put s.mem.db a (b.cast .fp32 c.dtype.partialType (s.db p lane)) }
  | _ => s

structure Reference (V : Type) where
  mem : Shared V
  dwReads : Nat → Nat → List V
  dbReads : Nat → Nat → List V

def Reference.initial (mem : Shared V) : Reference V := ⟨mem, fun _ _ => [], fun _ _ => []⟩

def referenceDW (b : Backend V) (rows : Nat → RowValues V) (r : Reference V) (p lane : Nat) : V :=
  (r.dwReads p lane).foldl b.add ((rows p).dw lane)
def referenceDB (b : Backend V) (rows : Nat → RowValues V) (r : Reference V) (p lane : Nat) : V :=
  (r.dbReads p lane).foldl b.add ((rows p).db lane)

/-- The reference has NO mutable FP accumulator. Shared loads are read from its
own memory and appended to observation lists; stores recompute ordered folds.
Even the first Count==0 store retains its storage cast and no artificial +0. -/
def observe (b : Backend V) (c : Config) (rows : Nat → RowValues V)
    (undef : Undef V) (e : Event) (r : Reference V) : Reference V :=
  match e with
  | .dx p lane a => { r with mem.dx := put r.mem.dx a ((rows p).dx lane) }
  | .addDW p lane a active =>
    let value := b.cast c.dtype.partialType .fp32 (if active then r.mem.dw a else undef .dw a)
    { r with dwReads := put r.dwReads p (put (r.dwReads p) lane (r.dwReads p lane ++ [value])) }
  | .addDB p lane a active =>
    let value := b.cast c.dtype.partialType .fp32 (if active then r.mem.db a else undef .db a)
    { r with dbReads := put r.dbReads p (put (r.dbReads p) lane (r.dbReads p lane ++ [value])) }
  | .storeDW p lane a => { r with mem.dw := put r.mem.dw a (b.cast .fp32 c.dtype.partialType (referenceDW b rows r p lane)) }
  | .storeDB p lane a => { r with mem.db := put r.mem.db a (b.cast .fp32 c.dtype.partialType (referenceDB b rows r p lane)) }
  | _ => r

def Related (b : Backend V) (rows : Nat → RowValues V) (s : Values V) (r : Reference V) : Prop :=
  s.mem = r.mem ∧ (∀ p lane, s.dw p lane = referenceDW b rows r p lane) ∧
    (∀ p lane, s.db p lane = referenceDB b rows r p lane)

theorem initial_related (b : Backend V) (rows : Nat → RowValues V) (mem : Shared V) :
    Related b rows (initialValues rows mem) (Reference.initial mem) :=
  ⟨rfl, fun _ _ => rfl, fun _ _ => rfl⟩

theorem execute_related (b : Backend V) (c : Config) (rows : Nat → RowValues V)
    (undef : Undef V) (e : Event) (s : Values V) (r : Reference V) (h : Related b rows s r) :
    Related b rows (execute b c rows undef e s) (observe b c rows undef e r) := by
  rcases h with ⟨hm, hdw, hdb⟩
  cases e <;> simp only [execute, observe, Related]
  all_goals first
    | exact ⟨hm, hdw, hdb⟩
    | skip
  case dx p lane a => exact ⟨by rw [hm], hdw, hdb⟩
  case storeDW p lane a => exact ⟨by rw [hm, hdw], hdw, hdb⟩
  case storeDB p lane a => exact ⟨by rw [hm, hdb], hdw, hdb⟩
  case addDW p lane a active =>
    refine ⟨hm, ?_, hdb⟩
    intro q j
    by_cases hp : q = p <;> by_cases hl : j = lane <;>
      simp [put, hp, hl, referenceDW, hdw, hm, List.foldl_append]
  case addDB p lane a active =>
    refine ⟨hm, hdw, ?_⟩
    intro q j
    by_cases hp : q = p <;> by_cases hl : j = lane <;>
      simp [put, hp, hl, referenceDB, hdb, hm, List.foldl_append]

def replay (b : Backend V) (c : Config) (rows : Nat → RowValues V) (undef : Undef V) :
    List Event → Reference V → Reference V
  | [], r => r
  | e :: es, r => replay b c rows undef es (observe b c rows undef e r)

structure State (V : Type) where
  control : Control
  values : Values V

def initial (rows : Nat → RowValues V) (mem : Shared V) (words : Int → I32) : State V :=
  ⟨initialControl words, initialValues rows mem⟩

def step (b : Backend V) (c : Config) (rows : Nat → RowValues V)
    (undef : Undef V) (p : Nat) (s : State V) : Event × State V :=
  let (e, next) := controlStep c p s.control
  (e, ⟨next, execute b c rows undef e s.values⟩)

def run (b : Backend V) (c : Config) (rows : Nat → RowValues V) (undef : Undef V) :
    List Nat → State V → List Event × State V
  | [], s => ([], s)
  | p :: ps, s =>
    let (e, next) := step b c rows undef p s
    let (events, finalState) := run b c rows undef ps next
    (e :: events, finalState)

theorem run_control_projection (b : Backend V) (c : Config) (rows : Nat → RowValues V)
    (undef : Undef V) (schedule : List Nat) (s : State V) :
    runControl c schedule s.control =
      ((run b c rows undef schedule s).1, (run b c rows undef schedule s).2.control) := by
  induction schedule generalizing s with
  | nil => rfl
  | cons p ps ih =>
    simp only [runControl, run, step]
    exact congrArg (fun result : List Event × Control => ((controlStep c p s.control).1 :: result.1, result.2))
      (ih ⟨(controlStep c p s.control).2, execute b c rows undef (controlStep c p s.control).1 s.values⟩)

theorem run_related (b : Backend V) (c : Config) (rows : Nat → RowValues V)
    (undef : Undef V) (schedule : List Nat) (s : State V) (r : Reference V)
    (h : Related b rows s.values r) :
    Related b rows (run b c rows undef schedule s).2.values
      (replay b c rows undef (run b c rows undef schedule s).1 r) := by
  induction schedule generalizing s r with
  | nil => exact h
  | cons p ps ih =>
    exact ih (step b c rows undef p s).2
      (observe b c rows undef (step b c rows undef p s).1 r)
      (execute_related b c rows undef (controlStep c p s.control).1 s.values r h)

/-- Whole arbitrary-schedule numerical contract with control-only reference
trace. No readiness, source-step equality, finite-value or final-result premise. -/
theorem whole_schedule (b : Backend V) (c : Config) (rows : Nat → RowValues V)
    (undef : Undef V) (schedule : List Nat) (mem : Shared V) (words : Int → I32) :
    (run b c rows undef schedule (initial rows mem words)).1 = controlTrace c schedule words ∧
    Related b rows (run b c rows undef schedule (initial rows mem words)).2.values
      (replay b c rows undef (controlTrace c schedule words) (Reference.initial mem)) := by
  have ht := congrArg Prod.fst (run_control_projection b c rows undef schedule (initial rows mem words))
  have hv := run_related b c rows undef schedule (initial rows mem words) (Reference.initial mem)
    (initial_related b rows mem)
  have he : (run b c rows undef schedule (initial rows mem words)).1 = controlTrace c schedule words := ht.symm
  exact ⟨he, by rw [← he]; exact hv⟩

#axiomsClean execute_related
#axiomsClean run_control_projection
#axiomsClean whole_schedule
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LayerNormLock
