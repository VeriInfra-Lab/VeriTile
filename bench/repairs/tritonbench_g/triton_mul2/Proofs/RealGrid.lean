import bench.tritonbench_g.triton_mul2.Proofs.Real
import bench.tritonbench_g.triton_mul2.Proofs.FiniteHost
import bench.tritonbench_g.vector_addition.Proofs.HostGrid

/-! Shared actual execution of both original source kernels. Variable-size
prefix composition runs each program on its predecessor memory and supports
logical in-place aliases. Existing per-program source proofs and ceil-grid
arithmetic are reused; no finite numerical model supplies a result. -/
namespace VeriTile.Bench.TritonBenchG.TritonMul2.RealGrid
open VeriTile.Triton
open TritonMul2Finite (Mode)
noncomputable section
set_option maxHeartbeats 4000000

def target : Mode → RegionName → RegionName → RegionName
  | .inplace, X, _ => X
  | .outplace, _, O => O

def source (mode : Mode) (X O : RegionName) (n B : Nat) : Kernel :=
  match mode with
  | .inplace => (mul2_inplace_kernel X n B).toAlgKernel
  | .outplace => (mul2_kernel X O n B).toAlgKernel

theorem source_body (mode : Mode) (X O : RegionName) (n B : Nat) :
    (source mode X O n B).body = (mul2_kernel X (target mode X O) n B).toAlgKernel.body := by
  cases mode <;> rfl

def programState (s : BlockState) (n B pid : Nat) : BlockState :=
  {s with
    regs := fun _ _ _ => none
    pids := fun ax => if ax = 0 then pid else 0
    numPids := fun ax => if ax = 0 then TritonMul2Finite.grid n B else 1}

def runPrefix (mode : Mode) (X O : RegionName) (n B : Nat) (initial : BlockState) : Nat → Option BlockState
  | 0 => some initial
  | count+1 => do
    let s ← runPrefix mode X O n B initial count
    exec (source mode X O n B) (programState s n B count)

def run (mode : Mode) (X O : RegionName) (n B : Nat) (initial : BlockState) : Option BlockState :=
  runPrefix mode X O n B initial (TritonMul2Finite.grid n B)

def Values (X O : RegionName) (n B count : Nat) (initial final : BlockState) : Prop :=
  ∀ o, o < n → o < count*B → final.readMem O o = 2*initial.readMem X o

def Frame (O : RegionName) (n B count : Nat) (initial final : BlockState) : Prop :=
  ∀ r o, r ≠ O ∨ n ≤ o ∨ count*B ≤ o → final.mem r o = initial.mem r o

theorem read_eq_of_mem {s t : BlockState} {r : RegionName} {o : Nat}
    (h : s.mem r o = t.mem r o) : s.readMem r o = t.readMem r o := by
  simp only [BlockState.readMem, h]

theorem program_correct (mode : Mode) (X O : RegionName) (n B pid : Nat) (s : BlockState) :
    ∃ t, exec (source mode X O n B) (programState s n B pid) = some t ∧
      (∀ o, o < n → pid*B ≤ o → o < (pid+1)*B →
        t.readMem (target mode X O) o = 2*s.readMem X o) ∧
      (∀ r o, r ≠ target mode X O ∨ n ≤ o ∨ o < pid*B ∨ (pid+1)*B ≤ o →
        t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hv, hf⟩ := mul2_kernel_region_run X (target mode X O) n B (programState s n B pid)
    (fun j => s.readMem X (pid*B+j.val)) (fun _ _ => rfl)
  refine ⟨t, ?_, ?_, ?_⟩
  · simpa only [exec, source_body] using he
  · intro o hn lower upper
    have upper' : o < pid*B+B := by simpa only [Nat.add_mul, Nat.one_mul] using upper
    let j : Fin B := ⟨o-pid*B, by omega⟩
    have hj : pid*B+j.val = o := by dsimp [j]; omega
    have value := hv j (by simpa [programState, BlockState.pid, hj] using hn)
    simpa [programState, BlockState.pid, hj] using value
  · intro r o outside
    apply hf r o
    rcases outside with different | beyond | before | after
    · exact Or.inl different
    · right; intro j active eq
      have a : pid*B+j.val < n := active
      have e : o = pid*B+j.val := eq
      omega
    · right; intro j _ eq
      have e : o = pid*B+j.val := eq
      omega
    · right; intro j _ eq
      have e : o = pid*B+j.val := eq
      have h := j.isLt
      have upper : pid*B+B ≤ o := by simpa only [Nat.add_mul, Nat.one_mul] using after
      omega

/-- The same variable-block induction handles outplace and inplace. Earlier
stores lie strictly before every later tile, so each tile still reads the
initial input values even when input and output logical regions coincide. -/
theorem prefix_correct (mode : Mode) (X O : RegionName) (n B count : Nat) (initial : BlockState) :
    ∃ final, runPrefix mode X O n B initial count = some final ∧
      Values X (target mode X O) n B count initial final ∧
      Frame (target mode X O) n B count initial final := by
  induction count with
  | zero => exact ⟨initial, rfl, by simp [Values], fun _ _ _ => rfl⟩
  | succ count ih =>
    obtain ⟨s, hs, hv, hf⟩ := ih
    obtain ⟨t, ht, tv, tf⟩ := program_correct mode X O n B count s
    refine ⟨t, by simp only [runPrefix,hs]; exact ht, ?_, ?_⟩
    · intro o hn bound
      by_cases old : o < count*B
      · exact (read_eq_of_mem (tf _ o (Or.inr (Or.inr (Or.inl old))))).trans (hv o hn old)
      · have lower : count*B ≤ o := by omega
        rw [tv o hn lower bound, read_eq_of_mem (hf X o (Or.inr (Or.inr lower)))]
    · intro r o outside
      have earlier : r ≠ target mode X O ∨ n ≤ o ∨ count*B ≤ o := by
        rcases outside with h | h | h
        · exact Or.inl h
        · exact Or.inr (Or.inl h)
        · right; right
          have upper : count*B+B ≤ o := by simpa only [Nat.succ_mul] using h
          omega
      have current : r ≠ target mode X O ∨ n ≤ o ∨ o < count*B ∨ (count+1)*B ≤ o := by
        rcases outside with h | h | h
        · exact Or.inl h
        · exact Or.inr (Or.inl h)
        · exact Or.inr (Or.inr (Or.inr h))
      exact (tf r o current).trans (hf r o earlier)

theorem full_grid_correct (mode : Mode) (X O : RegionName) (n B : Nat) (positive : 0 < B)
    (initial : BlockState) :
    ∃ final, run mode X O n B initial = some final ∧
      (∀ o, o < n → final.readMem (target mode X O) o = 2*initial.readMem X o) ∧
      (∀ r o, r ≠ target mode X O ∨ n ≤ o → final.mem r o = initial.mem r o) := by
  obtain ⟨t, he, hv, hf⟩ := prefix_correct mode X O n B (TritonMul2Finite.grid n B) initial
  have covers := VectorAddition.ValueProofs.grid_covers (⟨X,X,O,n,B⟩ : VectorAddition.ValueProofs.Parameters) positive
  change n ≤ TritonMul2Finite.grid n B * B at covers
  exact ⟨t,he,fun o ho => hv o ho (ho.trans_le covers), fun r o h => hf r o (by rcases h with h | h; exact Or.inl h; exact Or.inr (Or.inl h))⟩

/-- Signed source cdiv is retained until positivity is established. -/
theorem host_grid_positive (n B : Nat) (positive : 0 < B) :
    TritonMul2Finite.hostGrid n (B : Int) = (TritonMul2Finite.grid n B : Int) := by
  unfold TritonMul2Finite.hostGrid TritonMul2Finite.grid
  rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
  rw [Int.natCast_ediv, Int.natCast_sub (by omega : 1 ≤ n+B)]
  simp

#axiomsClean source_body
#axiomsClean program_correct
#axiomsClean prefix_correct
#axiomsClean full_grid_correct
#axiomsClean host_grid_positive
#specNonCircular Values avoiding [run, runPrefix, source, exec]
#specNonCircular Frame avoiding [run, runPrefix, source, exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.TritonMul2.RealGrid
