import bench.tritonbench_g.lora_expand_gemv.Proofs.HostProperties
import bench.tritonbench_g.lora_expand_gemv.BranchCorrect
namespace VeriTile.Bench.TritonBenchG.LoraExpandGemv.Host
open VeriTile.Triton FP.Structural
open FPValue (Word OutputFormat)
open FP.Equational (ReductionPlan)
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000
local instance hostGridWord : Inhabited Word := ⟨FP.Value.zero .fp32⟩

abbrev Memory := RegionName → Nat → Cell Word
def Launch.count (l : Launch) := l.parameters.outputSize * l.batches

/-- One complete grid traversal with pid0 varying fastest. Each program gets
fresh registers and the preceding program's actual memory. This is an explicit
logical sequential schedule, not a claim about native concurrent scheduling. -/
def programState (l : Launch) (initial : State Word) (mem : Memory) (k : Nat) : State Word :=
  { initial with
    mem := mem
    regs := fun _ _ _ => none
    pids := fun ax => if ax = 0 then k % l.parameters.outputSize
      else if ax = 1 then k / l.parameters.outputSize else 0
    numPids := fun ax => if ax = 0 then l.parameters.outputSize
      else if ax = 1 then l.batches else 1 }

def runPrefix (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (l : Launch) (out : OutputFormat) (initial : State Word) : Nat → Option (State Word)
  | 0 => some initial
  | k + 1 => (runPrefix rest cfg plans l out initial k).bind fun previous =>
      FP.Structural.exec (Branches.engine rest cfg plans)
        (Branches.source l.profile out l.parameters) (programState l initial previous.mem k)

def runLaunch (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (l : Launch) (out : OutputFormat) (initial : State Word) : Option (State Word) :=
  runPrefix rest cfg plans l out initial l.count

/-- Independent ordered memory recurrence. Each program's scalar/tree
reference reads the prior reference memory, so inter-program aliases and
sentinel-index changes are observed rather than merged from isolated runs. -/
def referencePrefix (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (l : Launch) (out : OutputFormat) (initial : State Word) : Nat → Memory
  | 0 => initial.mem
  | k + 1 => Branches.expectedMemory cfg plans l.profile out l.parameters
      (programState l initial (referencePrefix cfg plans l out initial k) k)

def expectedMemory (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (l : Launch) (out : OutputFormat) (initial : State Word) : Memory :=
  referencePrefix cfg plans l out initial l.count

theorem prefix_correct (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (l : Launch) (out : OutputFormat) (initial : State Word)
    (hBN : 0 < l.parameters.blockRows) (k : Nat) :
    ∃ final, runPrefix rest cfg plans l out initial k = some final ∧
      final.mem = referencePrefix cfg plans l out initial k := by
  induction k with
  | zero => exact ⟨initial, rfl, rfl⟩
  | succ k ih =>
    obtain ⟨previous, hr, hm⟩ := ih
    obtain ⟨final, he, hmem, _⟩ :=
      BranchCorrect.lora_expand_all_branches_machine_correctness rest cfg plans l.profile out
        l.parameters (programState l initial previous.mem k) hBN
    refine ⟨final, ?_, ?_⟩
    · simp only [runPrefix, hr, Option.bind_some]
      exact he
    · rw [hmem, hm]
      rfl

def Outside (l : Launch) (r : RegionName) (o : Nat) : Prop :=
  r ≠ l.parameters.output ∨
    ∀ x : Fin l.parameters.outputSize, ∀ b : Fin l.batches,
      ∀ m : Fin (Branches.length l.parameters),
        o ≠ b.val * l.parameters.outputBatchStride + x.val * Branches.length l.parameters +
          m.val * l.parameters.outputStride

theorem coordinate_bounds (l : Launch) (k : Nat) (hk : k < l.count) :
    k % l.parameters.outputSize < l.parameters.outputSize ∧
      k / l.parameters.outputSize < l.batches := by
  have hn : 0 < l.parameters.outputSize := by
    unfold Launch.count at hk
    by_contra h
    have hz : l.parameters.outputSize = 0 := by omega
    simp [hz] at hk
  refine ⟨Nat.mod_lt _ hn, ?_⟩
  apply (Nat.div_lt_iff_lt_mul hn).2
  simpa [Launch.count, Nat.mul_comm] using hk

theorem coordinates_cover (l : Launch) (x : Fin l.parameters.outputSize) (b : Fin l.batches) :
    b.val * l.parameters.outputSize + x.val < l.count ∧
      (b.val * l.parameters.outputSize + x.val) % l.parameters.outputSize = x.val ∧
      (b.val * l.parameters.outputSize + x.val) / l.parameters.outputSize = b.val := by
  have hn : 0 < l.parameters.outputSize := Nat.zero_lt_of_lt x.isLt
  have bound := Nat.mul_le_mul_right l.parameters.outputSize (Nat.succ_le_of_lt b.isLt)
  rw [Nat.succ_mul] at bound
  have hx := x.isLt
  have hi : b.val * l.parameters.outputSize + x.val < l.count := by
    unfold Launch.count
    rw [Nat.mul_comm l.parameters.outputSize l.batches]
    omega
  refine ⟨hi, ?_, ?_⟩
  · simp [Nat.add_mod, Nat.mod_eq_of_lt x.isLt]
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hn, Nat.div_eq_of_lt x.isLt]
    omega

theorem outside_program (l : Launch) (initial : State Word) (mem : Memory)
    (k : Nat) (hk : k < l.count) (r : RegionName) (o : Nat) (h : Outside l r o) :
    Branches.Outside l.parameters (programState l initial mem k) r o := by
  rcases h with h | h
  · exact Or.inl h
  · right
    intro m
    obtain ⟨hx,hb⟩ := coordinate_bounds l k hk
    simpa [Branches.outputOffset, Branches.outputBase, programState] using
      h ⟨k % l.parameters.outputSize,hx⟩ ⟨k / l.parameters.outputSize,hb⟩ m

theorem reference_frame (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (l : Launch) (out : OutputFormat) (initial : State Word)
    (k : Nat) (hk : k ≤ l.count) (r : RegionName) (o : Nat) (ho : Outside l r o) :
    referencePrefix cfg plans l out initial k r o = initial.mem r o := by
  induction k with
  | zero => rfl
  | succ k ih =>
    rw [referencePrefix, Branches.expectedMemory_frame cfg plans l.profile out l.parameters
      (programState l initial (referencePrefix cfg plans l out initial k) k) r o
      (outside_program l initial _ k (by omega) r o ho)]
    exact ih (by omega)

theorem launch_correct (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (l : Launch) (out : OutputFormat) (initial : State Word)
    (hBN : 0 < l.parameters.blockRows) :
    ∃ final, runLaunch rest cfg plans l out initial = some final ∧
      final.mem = expectedMemory cfg plans l out initial ∧
      (∀ r o, Outside l r o → final.mem r o = initial.mem r o) := by
  obtain ⟨final, he, hm⟩ := prefix_correct rest cfg plans l out initial hBN l.count
  refine ⟨final, he, hm, ?_⟩
  intro r o ho
  rw [hm]
  exact reference_frame cfg plans l out initial l.count (by rfl) r o ho

#specNonCircular expectedMemory avoiding [runLaunch, runPrefix, FP.Structural.exec, Branches.source]
#axiomsClean prefix_correct
#axiomsClean coordinate_bounds
#axiomsClean coordinates_cover
#axiomsClean outside_program
#axiomsClean reference_frame
#axiomsClean launch_correct
end
end VeriTile.Bench.TritonBenchG.LoraExpandGemv.Host
