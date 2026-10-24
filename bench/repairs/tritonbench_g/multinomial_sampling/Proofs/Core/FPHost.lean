import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPComplete

/-! Logical host launch for the original multinomial kernel. Every program
executes the admitted source; the final reference reads only the initial
scores/metadata. Fresh output storage is the clone allocation's separation
condition. Native allocation, launch scheduling, and integer address lowering
are external bindings, not numerical hypotheses of this theorem. -/
namespace VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

def FreshOutput (p : Parameters) : Prop :=
  p.scores ≠ p.output ∧ p.seeds ≠ p.output ∧
  p.offsets ≠ p.output ∧ p.indices ≠ p.output

def grid (p : Parameters) : Nat := (p.batches + p.block - 1) / p.block

/-- Program-local registers are fresh. Undefined masked-load values are held
fixed across the launch, but may be arbitrary signed64 values in SeedDomain. -/
def programState (p : Parameters) (initial previous : State α) (pid : Nat) : State α :=
  { initial with
    mem := previous.mem
    regs := fun _ _ _ => none
    pids := fun a => if a = 0 then pid else 0
    numPids := fun a => if a = 0 then grid p else 1 }

def InputsEqual (p : Parameters) (a b : State α) : Prop :=
  a.pids = b.pids ∧ a.undef = b.undef ∧
  a.mem p.scores = b.mem p.scores ∧ a.mem p.seeds = b.mem p.seeds ∧
  a.mem p.offsets = b.mem p.offsets ∧ a.mem p.indices = b.mem p.indices

theorem seedDomain_congr (p : Parameters) (a b : State α) (h : InputsEqual p a b) :
    SeedDomain p a = SeedDomain p b := by
  rcases h with ⟨hp, hu, _, hs, _, _⟩
  simp only [SeedDomain, seedValue, intLoad, active, row, hp, hu, hs]
  rfl

theorem rounds_congr (P : Primitives α) (d : Storage) (p : Parameters)
    (a b : State α) (h : InputsEqual p a b) (k : Nat) :
    rounds P d p a k = rounds P d p b k := by
  rcases h with ⟨hp, hu, hsc, hse, ho, hi⟩
  have heScores (n : Nat) : scores P d p a n = scores P d p b n := by
    funext i
    simp only [scores, active, row, scoreOffset, hp, hsc]
  have heSample : sample P p a = sample P p b := by
    funext i
    simp only [sample, seedValue, offsetValue, intLoad, active, row, hp, hu, hse, ho]
    rfl
  have heValid (n : Nat) (st : Accumulator α p) : valid P d p a n st = valid P d p b n st := by
    funext i
    simp only [valid, cumulative, scanned, heScores, heSample]
  induction k with
  | zero =>
    simp only [rounds, initialAccumulator]
    congr 1
    funext i
    simp only [initialOutput, intLoad, active, row, hp, hu, hi]
    rfl
  | succ k ih =>
    simp only [rounds, ih, advance, selectedIndex, selected, found, valid,
      cumulative, scanned, heScores, heSample, heValid, active, row, hp, hi]
    rfl

def runPrefix (rest : Algebra α) (P : Primitives α) (d : Storage)
    (p : Parameters) (initial : State α) : Nat → Option (State α)
  | 0 => some initial
  | k + 1 => (runPrefix rest P d p initial k).bind fun previous =>
      FP.Structural.exec (backend rest P) (source d p) (programState p initial previous k)

def run (rest : Algebra α) (P : Primitives α) (d : Storage)
    (p : Parameters) (initial : State α) : Option (State α) :=
  runPrefix rest P d p initial (grid p)

/-- Unique program and local lane for a global batch row. -/
def lane (p : Parameters) (o : Nat) : Row p :=
  (⟨o % p.block, Nat.mod_lt _ p.blockDomain.1⟩, PUnit.unit)

def value (P : Primitives α) (d : Storage) (p : Parameters)
    (initial : State α) (o : Nat) : Int :=
  (rounds P d p (programState p initial initial (o / p.block)) (count p)).output (lane p o)

def expectedMemory (P : Primitives α) (d : Storage) (p : Parameters)
    (initial : State α) : RegionName → Nat → Cell α :=
  fun r o => if r = p.output ∧ o < p.batches then .mk .int (value P d p initial o)
    else initial.mem r o

private theorem block_value (P : Primitives α) (d : Storage) (p : Parameters)
    (initial previous : State α) (k : Nat)
    (same : InputsEqual p (programState p initial previous k) (programState p initial initial k))
    (i : Row p) :
    (rounds P d p (programState p initial previous k) (count p)).output i =
      value P d p initial (k * p.block + i.1.val) := by
  rw [rounds_congr P d p _ _ same]
  have hd : (k * p.block + i.1.val) / p.block = k := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ p.blockDomain.1,
      Nat.div_eq_of_lt i.1.isLt, Nat.zero_add]
  have hm : (k * p.block + i.1.val) % p.block = i.1.val := by
    simp [Nat.add_mod, Nat.mod_eq_of_lt i.1.isLt]
  have hl : lane p (k * p.block + i.1.val) = i := by
    apply Prod.ext
    · apply Fin.ext; exact hm
    · exact Subsingleton.elim _ _
  simp only [value, hd, hl]

/-- Successful execution of every prefix, with exact global output and frame.
SeedDomain is the existing signed64 storage requirement, not a correctness
assumption about a program's result. -/
theorem prefix_correct (rest : Algebra α) (P : Primitives α) (d : Storage)
    (p : Parameters) (initial : State α) (fresh : FreshOutput p)
    (seeds : ∀ k, k < grid p → SeedDomain p (programState p initial initial k))
    (k : Nat) (hk : k ≤ grid p) :
    ∃ final, runPrefix rest P d p initial k = some final ∧
      ∀ r o, final.mem r o =
        if r = p.output ∧ o < p.batches ∧ o < k * p.block
        then .mk .int (value P d p initial o) else initial.mem r o := by
  induction k with
  | zero => exact ⟨initial, rfl, by intro r o; simp⟩
  | succ k ih =>
    have hsucc : (k + 1) * p.block = k * p.block + p.block := by simp [Nat.add_mul]
    obtain ⟨previous, hr, hm⟩ := ih (by omega)
    let start := programState p initial previous k
    have input_eq (r : RegionName) (h : r ≠ p.output) : previous.mem r = initial.mem r := by
      funext o
      simpa only [h, false_and, ↓reduceIte] using hm r o
    have same : InputsEqual p start (programState p initial initial k) :=
      ⟨rfl, rfl, input_eq p.scores fresh.1, input_eq p.seeds fresh.2.1,
        input_eq p.offsets fresh.2.2.1, input_eq p.indices fresh.2.2.2⟩
    have hs : SeedDomain p start := by
      rw [seedDomain_congr p _ _ same]
      exact seeds k (by omega)
    obtain ⟨final, he, _, hv, hf⟩ := correctness rest P d p start hs
    refine ⟨final, ?_, ?_⟩
    · simp only [runPrefix, hr, Option.bind_some]
      exact he
    · intro r o
      by_cases here : r = p.output ∧ k * p.block ≤ o ∧ o < (k + 1) * p.block ∧ o < p.batches
      · obtain ⟨hr, hlo, hhi, hn⟩ := here
        subst r
        let i : Row p := (⟨o - k * p.block, by omega⟩, PUnit.unit)
        have hoff : row p start i = o := by dsimp [row, start, programState, i]; omega
        have ha : active p start i := by change row p start i < p.batches; rwa [hoff]
        have hout := hv i ha
        rw [hoff] at hout
        rw [hout]
        have he := block_value P d p initial previous k same i
        have hi : k * p.block + i.1.val = o := by dsimp [i]; omega
        rw [hi] at he
        rw [he]
        simp only [hn, hhi, and_self, ↓reduceIte]
      · have untouched : r ≠ p.output ∨ ∀ i : Row p, active p start i → o ≠ row p start i := by
          by_cases hr : r = p.output
          · right
            intro i ha he
            have hi := i.1.isLt
            dsimp [active, row, start, programState] at ha he
            apply here
            refine ⟨hr, ?_, ?_, ?_⟩ <;> omega
          · exact Or.inl hr
        rw [hf r o untouched]
        change previous.mem r o = _
        rw [hm]
        congr 1
        apply propext
        constructor
        · rintro ⟨hr, hn, hk⟩; exact ⟨hr, hn, by omega⟩
        · rintro ⟨hr, hn, hk⟩
          refine ⟨hr, hn, ?_⟩
          by_contra old
          apply here
          exact ⟨hr, by omega, hk, hn⟩

theorem grid_covers (p : Parameters) : p.batches ≤ grid p * p.block := by
  have h := p.blockDomain.1
  have hd := Nat.mod_lt (p.batches + p.block - 1) h
  have he := Nat.mod_add_div (p.batches + p.block - 1) p.block
  rw [Nat.mul_comm p.block] at he
  unfold grid
  omega

theorem launch_correct (rest : Algebra α) (P : Primitives α) (d : Storage)
    (p : Parameters) (initial : State α) (fresh : FreshOutput p)
    (seeds : ∀ k, k < grid p → SeedDomain p (programState p initial initial k)) :
    ∃ final, run rest P d p initial = some final ∧
      final.mem = expectedMemory P d p initial := by
  obtain ⟨final, he, hm⟩ := prefix_correct rest P d p initial fresh seeds (grid p) (by rfl)
  refine ⟨final, he, ?_⟩
  funext r o
  rw [hm]
  unfold expectedMemory
  have hc := grid_covers p
  by_cases active : r = p.output ∧ o < p.batches
  · have hg : o < grid p * p.block := by omega
    simp only [active.1, active.2, hg, and_self, ↓reduceIte]
  · have hn : ¬ (r = p.output ∧ o < p.batches ∧ o < grid p * p.block) := by tauto
    simp only [active, hn, ↓reduceIte]

#specNonCircular expectedMemory avoiding [run, runPrefix, FP.Structural.exec, source]
#axiomsClean rounds_congr
#axiomsClean prefix_correct
#axiomsClean launch_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host
