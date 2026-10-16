import bench.tritonbench_g.layer_norm_triton.Proofs.BackwardValues

namespace VeriTile.Bench.TritonBenchG.LayerNormLock

def lockAddr (c : Config) (p : Nat) : Int := (BitVec.ofNat 32 p |>.srem c.group).toInt
def countAddr (c : Config) (p : Nat) : Int := lockAddr c p + c.group.toInt
/-- DW + lock_id*N + cols uses two separately sign-extended addptr offsets. -/
def partialAddr (c : Config) (p lane : Nat) : Int :=
  ((BitVec.ofNat 32 p |>.srem c.group) * c.n).toInt + (BitVec.ofNat 32 lane).toInt

def groupDefined (c : Config) (p : Nat) : Bool :=
  c.group != 0 && !(BitVec.ofNat 32 p == 0x80000000 && c.group == 0xffffffff)

inductive Phase where
  | prepare | dx (lane : Nat) | poll | count | setCount
  | loadDW (lane : Nat) | loadDB (lane : Nat)
  | storeDW (lane : Nat) | storeDB (lane : Nat)
  | release | done | fault
  deriving DecidableEq, Repr

inductive Event where
  | silent
  | prepare (pid : Nat)
  | dx (pid lane : Nat) (address : Int)
  | poll (address : Int)
  | readCount (address : Int)
  | setCount (address : Int)
  | addDW (pid lane : Nat) (address : Int) (live : Bool)
  | addDB (pid lane : Nat) (address : Int) (live : Bool)
  | storeDW (pid lane : Nat) (address : Int)
  | storeDB (pid lane : Nat) (address : Int)
  | release (address : Int)
  deriving DecidableEq, Repr

/-- Exact CAS(0,1): old returned value controls the ==1 retry, not !=0. -/
def cas01 (old : I32) : I32 × I32 := (old, if old == 0 then 1 else old)

def integerEffect (e : Event) (words : Int → I32) : Int → I32 :=
  match e with
  | .poll a => put words a (cas01 (words a)).2
  | .setCount a => put words a 1
  | .release a => put words a 0
  | _ => words

structure Control where
  words : Int → I32
  pc : Nat → Phase

def initialControl (words : Int → I32) : Control := ⟨words, fun _ => .prepare⟩

/-- Every full-source instruction/control phase is represented. False-mask
loads retain undefined values in the arithmetic event; only stores disappear. -/
def decode (c : Config) (p : Nat) (s : Control) : Event × Phase :=
  let phase := s.pc p
  if p ≥ c.grid then (.silent, phase) else
  match phase with
  | .prepare => if groupDefined c p then (.prepare p, .dx 0) else (.silent, .fault)
  | .dx lane =>
    if lane < c.block then
      (if live c 0 lane then .dx p lane (rowAddr c p (BitVec.ofNat 32 lane)) else .silent, .dx (lane+1))
    else (.silent, .poll)
  | .poll => (.poll (lockAddr c p), if s.words (lockAddr c p) == 1 then .poll else .count)
  | .count => (.readCount (countAddr c p), if s.words (countAddr c p) == 0 then .setCount else .loadDW 0)
  | .setCount => (.setCount (countAddr c p), .storeDW 0)
  | .loadDW lane =>
    if lane < c.block then (.addDW p lane (partialAddr c p lane) (live c 0 lane), .loadDW (lane+1))
    else (.silent, .loadDB 0)
  | .loadDB lane =>
    if lane < c.block then (.addDB p lane (partialAddr c p lane) (live c 0 lane), .loadDB (lane+1))
    else (.silent, .storeDW 0)
  | .storeDW lane =>
    if lane < c.block then
      (if live c 0 lane then .storeDW p lane (partialAddr c p lane) else .silent, .storeDW (lane+1))
    else (.silent, .storeDB 0)
  | .storeDB lane =>
    if lane < c.block then
      (if live c 0 lane then .storeDB p lane (partialAddr c p lane) else .silent, .storeDB (lane+1))
    else (.silent, .release)
  | .release => (.release (lockAddr c p), .done)
  | .done | .fault => (.silent, phase)

def controlStep (c : Config) (p : Nat) (s : Control) : Event × Control :=
  let (e, phase) := decode c p s
  (e, ⟨integerEffect e s.words, put s.pc p phase⟩)

def runControl (c : Config) : List Nat → Control → List Event × Control
  | [], s => ([], s)
  | p :: ps, s =>
    let (e, next) := controlStep c p s
    let (events, finalState) := runControl c ps next
    (e :: events, finalState)

def controlTrace (c : Config) (schedule : List Nat) (words : Int → I32) : List Event :=
  (runControl c schedule (initialControl words)).1

/-- Starting from the actual zero-initialized allocation, every word remains
0/1. This includes Count and does not presume lock availability at each CAS. -/
def BinaryWords (words : Int → I32) : Prop := ∀ a, words a = 0 ∨ words a = 1

theorem integer_effect_binary (e : Event) (words : Int → I32) (h : BinaryWords words) :
    BinaryWords (integerEffect e words) := by
  have hp (key : Int) (value : I32) (hv : value = 0 ∨ value = 1) :
      BinaryWords (put words key value) := by
    intro a
    by_cases ha : a = key
    · simp only [put, if_pos ha]; exact hv
    · simp only [put, if_neg ha]; exact h a
  cases e <;> simp only [integerEffect]
  all_goals try exact h
  case poll address =>
    apply hp
    rcases h address with hz | ho
    · simp [cas01, hz]
    · simp [cas01, ho]
  case setCount address => exact hp address 1 (Or.inr rfl)
  case release address => exact hp address 0 (Or.inl rfl)

theorem control_step_binary (c : Config) (p : Nat) (s : Control) (h : BinaryWords s.words) :
    BinaryWords (controlStep c p s).2.words := by
  exact integer_effect_binary (decode c p s).1 s.words h

theorem control_run_binary (c : Config) (schedule : List Nat) (s : Control)
    (h : BinaryWords s.words) : BinaryWords (runControl c schedule s).2.words := by
  induction schedule generalizing s with
  | nil => exact h
  | cons p ps ih => exact ih (controlStep c p s).2 (control_step_binary c p s h)

theorem initial_zero_binary (c : Config) (schedule : List Nat) :
    BinaryWords (runControl c schedule (initialControl (fun _ => 0))).2.words :=
  control_run_binary c schedule _ (fun _ => Or.inl rfl)

theorem binary_cas_exit_is_acquire (old : I32) (h : old = 0 ∨ old = 1)
    (exitLoop : (old == 1) = false) : old = 0 ∧ (cas01 old).2 = 1 := by
  rcases h with h | h
  · subst old; decide
  · subst old; contradiction

/-- An invalid lock word 2 exits the actual while without acquiring it. Never
replace source ==1 by a generic spin-until-success rule for arbitrary words. -/
theorem invalid_word_two_exits_without_acquisition :
    ((2#32) == 1) = false ∧ cas01 2 = (2, 2) := by decide +kernel

#axiomsClean integer_effect_binary
#axiomsClean initial_zero_binary
#axiomsClean binary_cas_exit_is_acquire
#axiomsClean invalid_word_two_exits_without_acquisition
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LayerNormLock
