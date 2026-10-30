import bench.repairs.tritonbench_g.square_matrix.Proofs.Contract
import VeriTile.Triton.Float.Structural
import bench.tritonbench_g.vector_addition.Proofs.Value

namespace VeriTile.Bench.Repairs.SquareMatrix.FPProof
open VeriTile.Triton FP.Structural
noncomputable section
open Classical
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Source FP32 precision is the ComputeOp profile; the structural .real tag
is an opaque carrier and is never interpreted as mathematical Real arithmetic. -/
def backend {α : Type} (M : Algebra α) : Algebra α := {M with fp32Load := id}

def reference {α : Type} (M : Algebra α) (x : α) : α :=
  M.binary (some .fp32) .real .mul x x

def execution {α : Type} [Inhabited α] (M : Algebra α)
    (n B : Nat) (s : State α) :
    {t : State α // FP.Structural.exec (backend M) (fp32Kernel n B) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [fp32Kernel, VeriTile.Bench.TritonBenchG.SquareMatrix.typedKernel,
    FP.Structural.exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, numericLt, natLt, bop, store, ComputeDType.eraseDType, numeric, backend]
  rfl


def offset {α : Type} (n : Nat) (s : State α) (i : TileIndex [B]) : Nat := s.pids 0*n+i.1.val

def resultCell {α : Type} [Inhabited α] (M : Algebra α)
    (mem : RegionName → Nat → Cell α) (i : Nat) : Cell α :=
  .mk .real (reference M ((mem "x" i).read .real))

def expectedMemory {α : Type} [Inhabited α] (M : Algebra α)
    (n B : Nat) (s : State α) : RegionName → Nat → Cell α :=
  (TileShape.allIndices [B]).foldl (fun mem i =>
    if i.1.val < n then fun r o =>
      if r = "out" ∧ o = offset n s i then resultCell M s.mem (offset n s i) else mem r o
    else mem) s.mem

theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α)
    (n B : Nat) (s : State α) :
    (execution M n B s).val.mem = expectedMemory M n B s := by
  simp [execution, expectedMemory, offset, resultCell, reference, bop,
    Broadcast.leftIndex, Broadcast.rightIndex,
    VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.masked_scatter_memory,
    State.setReg, numeric, backend]
  congr 1
  funext mem i
  by_cases hi : i.1.val < n
  · simp only [hi, ↓reduceIte]
  · simp only [hi, ↓reduceIte]

theorem expected_scatter {α : Type} [Inhabited α] (M : Algebra α)
    (n B : Nat) (s : State α) :
    expectedMemory M n B s =
      ((TileShape.allIndices [B]).foldl (fun t i =>
        if i.1.val < n then t.write "out" (offset n s i)
          (resultCell M s.mem (offset n s i)) else t) s).mem := by
  rw [VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.masked_scatter_memory]
  rfl

theorem offset_injective {α : Type} (n : Nat) (s : State α) :
    Function.Injective (offset (B:=B) n s) := by
  intro i j h
  rcases i with ⟨i,u⟩
  rcases j with ⟨j,v⟩
  have hv : i.val=j.val := by simpa [offset] using h
  have hi : i=j := Fin.ext hv
  subst j
  rfl

theorem expected_readback {α : Type} [Inhabited α] (M : Algebra α)
    (n B : Nat) (s : State α) (i : TileIndex [B]) (hi : i.1.val < n) :
    expectedMemory M n B s "out" (offset n s i) = resultCell M s.mem (offset n s i) := by
  rw [expected_scatter]
  exact State.masked_scatter_readback "out" (offset n s)
    (fun i => resultCell M s.mem (offset n s i)) (fun i => i.1.val < n) s i hi (offset_injective n s)

theorem expected_frame {α : Type} [Inhabited α] (M : Algebra α)
    (n B : Nat) (s : State α) (r : RegionName) (o : Nat)
    (h : r ≠ "out" ∨ ∀ i : TileIndex [B], i.1.val < n → offset n s i ≠ o) :
    expectedMemory M n B s r o=s.mem r o := by
  rw [expected_scatter]
  apply State.masked_scatter_frame
  rcases h with hr | ho
  · exact Or.inl hr
  · exact Or.inr (fun i _ hi => Ne.symm (ho i hi))

theorem expected_cell {α : Type} [Inhabited α] (M : Algebra α)
    (n B : Nat) (s : State α) (cover : n ≤ B) (r : RegionName) (o : Nat) :
    expectedMemory M n B s r o =
      if r="out" ∧ s.pids 0*n ≤ o ∧ o < s.pids 0*n+n then resultCell M s.mem o else s.mem r o := by
  by_cases h : r="out" ∧ s.pids 0*n ≤ o ∧ o < s.pids 0*n+n
  · rw [if_pos h]
    let i : TileIndex [B] := (⟨o-s.pids 0*n, by omega⟩,PUnit.unit)
    have ho : offset n s i=o := by simp only [offset,i]; omega
    rw [h.1,←ho]
    exact expected_readback M n B s i (by dsimp [i]; omega)
  · rw [if_neg h]
    apply expected_frame
    by_cases hr : r="out"
    · right
      intro i hi ho
      apply h
      unfold offset at ho
      exact ⟨hr,by omega,by omega⟩
    · exact Or.inl hr

theorem complete_correct {α : Type} [Inhabited α] (M : Algebra α)
    (n B : Nat) (s : State α) :
    ∃ t, FP.Structural.exec (backend M) (fp32Kernel n B) s=some t ∧
      t.mem=expectedMemory M n B s :=
  ⟨(execution M n B s).val,(execution M n B s).property,execution_memory M n B s⟩

def programState {α : Type} (initial : State α) (memory : RegionName → Nat → Cell α)
    (rows row : Nat) : State α :=
  { initial with
    mem := memory
    pids := fun ax => if ax = 0 then row else 0
    numPids := fun ax => if ax = 0 then rows else 1 }

def runPrefix {α : Type} [Inhabited α] (M : Algebra α)
    (n B rows : Nat) (s : State α) : Nat → Option (State α)
  | 0 => some s
  | k+1 => (runPrefix M n B rows s k).bind fun t =>
      FP.Structural.exec (backend M) (fp32Kernel n B) (programState s t.mem rows k)

def prefixMemory {α : Type} [Inhabited α] (M : Algebra α)
    (n : Nat) (s : State α) (k : Nat) : RegionName → Nat → Cell α :=
  fun r o => if r="out" ∧ o < k*n then resultCell M s.mem o else s.mem r o

theorem reference_step {α : Type} [Inhabited α] (M : Algebra α)
    (n B rows : Nat) (s : State α) (k : Nat) (cover : n ≤ B) :
    expectedMemory M n B (programState s (prefixMemory M n s k) rows k) =
      prefixMemory M n s (k+1) := by
  funext r o
  rw [expected_cell M n B _ cover]
  simp only [programState,↓reduceIte]
  have inputs : resultCell M (prefixMemory M n s k) o = resultCell M s.mem o := by
    simp [resultCell,prefixMemory]
  rw [inputs]
  by_cases before : o < k*n
  · have later : o < (k+1)*n := by rw [Nat.add_mul]; omega
    simp [prefixMemory,before,later,Nat.not_le.mpr before]
  · have lower : k*n ≤ o := Nat.le_of_not_gt before
    simp [prefixMemory,before,lower,Nat.add_mul]

theorem prefix_correct {α : Type} [Inhabited α] (M : Algebra α)
    (n B rows : Nat) (s : State α) (k : Nat) (cover : n ≤ B) :
    ∃ t, runPrefix M n B rows s k=some t ∧ t.mem=prefixMemory M n s k := by
  induction k with
  | zero => exact ⟨s,rfl,by funext r i; simp [prefixMemory]⟩
  | succ k ih =>
    obtain ⟨previous,hr,hm⟩ := ih
    obtain ⟨final,he,hf⟩ := complete_correct M n B (programState s previous.mem rows k)
    refine ⟨final,?_,?_⟩
    · simp only [runPrefix,hr,Option.bind_some]; exact he
    · rw [hf,hm,reference_step M n B rows s k cover]

#specNonCircular reference avoiding [fp32Kernel,execution,runPrefix]
#axiomsClean complete_correct
#axiomsClean prefix_correct
#guard_msgs (drop info) in
#auditModuleAxioms

end
end VeriTile.Bench.Repairs.SquareMatrix.FPProof
