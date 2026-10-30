import bench.tritonbench_g.mean_reduction.Proofs.SourceTensor
import bench.tritonbench_g.mean_reduction.Proofs.WholeGrid
import bench.tritonbench_g.mean_reduction.Proofs.WholeShape
import bench.tritonbench_g.mean_reduction.Proofs.WholeFP

namespace VeriTile.Bench.TritonBenchG.MeanReduction.Whole
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000

/-- Ordinary permute+contiguous semantics: exact coordinate copying, the actual
source ax order, and preservation of preexisting words. The copied values
are original input values, never precomputed means. -/
structure Contiguous (α : Type) (x : Tensor rank) (dims : List (Fin rank))
    (nodup : dims.Nodup) (entry : Nat → α) (preexisting : Nat → Prop) where
  base : Nat
  memory : Nat → α
  copy : ∀ (row : Fin (rows x.shape dims)) (col : Fin (columns x.shape dims)),
    memory (base + row.val*columns x.shape dims+col.val) =
      compressedValue x.shape dims nodup x.stride (x.read entry) row col
  frame : ∀ a, ((∀ i, i < rows x.shape dims*columns x.shape dims → a ≠ base+i) ∨ preexisting a) →
    memory a = entry a

def Contiguous.shape (c : Contiguous α x dims nodup entry preexisting) : List Nat :=
  ((batchAxes dims) ++ sortedReduction dims x.stride).map x.shape

theorem contiguous_numel (c : Contiguous α x dims nodup entry preexisting) :
    c.shape.prod = rows x.shape dims*columns x.shape dims := by
  simp only [Contiguous.shape,List.map_append,List.prod_append,sorted_columns,rows]

/-- Ordinary empty(shape, FP32, same-device) allocation: arbitrary output
initial contents, fresh output words and the full exterior allocation frame. -/
structure Allocated (c : Contiguous α x dims nodup entry preexisting) where
  base : Nat
  memory : Nat → α
  fresh : ∀ i, i < rows x.shape dims → ¬ preexisting (base+i)
  separate : ∀ i, i < rows x.shape dims → ∀ j,
    j < rows x.shape dims*columns x.shape dims → base+i ≠ c.base+j
  frame : ∀ a, (∀ i, i < rows x.shape dims → a ≠ base+i) → memory a = c.memory a

variable {α : Type} {rank : Nat} {x : Tensor rank} {dims : List (Fin rank)}
  {nodup : dims.Nodup} {entry : Nat → α} {preexisting : Nat → Prop}
  {c : Contiguous α x dims nodup entry preexisting}

theorem Allocated.input_copy (a : Allocated c)
    (row : Fin (rows x.shape dims)) (col : Fin (columns x.shape dims)) :
    a.memory (c.base+row.val*columns x.shape dims+col.val) =
      compressedValue x.shape dims nodup x.stride (x.read entry) row col := by
  rw [a.frame _ (by
    intro i hi same
    apply a.separate i hi (row.val*columns x.shape dims+col.val)
    · nlinarith [row.isLt,col.isLt]
    · simpa only [Nat.add_assoc] using same.symm)]
  exact c.copy row col

theorem Allocated.protected_frame (a : Allocated c) (addr : Nat) (hp : preexisting addr) :
    a.memory addr = entry addr := by
  rw [a.frame addr (by
    intro i hi equal
    apply a.fresh i hi
    simpa only [← equal] using hp)]
  exact c.frame addr (Or.inr hp)

/-- Every output-region address is committed, including addresses beyond the
output extent; the source frame proves such addresses retain their old value.
Thus this is not a truncating postprocessor that could hide invalid writes. -/
def Allocated.commit (a : Allocated c) (readOut : Nat → α) : Nat → α :=
  fun addr => if a.base ≤ addr then readOut (addr-a.base) else a.memory addr

theorem Allocated.commit_value (a : Allocated c) (readOut : Nat → α) (i : Nat) :
    a.commit readOut (a.base+i) = readOut i := by simp [Allocated.commit]

theorem Allocated.commit_frame (a : Allocated c) (readOut : Nat → α)
    (frame : ∀ i, rows x.shape dims ≤ i → readOut i = a.memory (a.base+i))
    (addr : Nat) (outside : ∀ i, i < rows x.shape dims → addr ≠ a.base+i) :
    a.commit readOut addr = a.memory addr := by
  unfold Allocated.commit
  split
  · rename_i hb
    have hi : rows x.shape dims ≤ addr-a.base := by
      by_contra h
      exact outside (addr-a.base) (by omega) (by omega)
    rw [frame _ hi]
    congr 1
    omega
  · rfl

def Allocated.realState {entry : Nat → ℝ} {c : Contiguous ℝ x dims nodup entry preexisting}
    (a : Allocated c) : BlockState where
  mem := fun r i => MemCell.real (if r = "X" then a.memory (c.base+i) else a.memory (a.base+i))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun ax => if ax = 0 then gridSize (rows x.shape dims) else 1

@[simp] theorem realState_input {entry : Nat → ℝ} {c : Contiguous ℝ x dims nodup entry preexisting}
    (a : Allocated c) (i : Nat) : a.realState.readMem "X" i = a.memory (c.base+i) := rfl

@[simp] theorem realState_output {entry : Nat → ℝ} {c : Contiguous ℝ x dims nodup entry preexisting}
    (a : Allocated c) (i : Nat) : a.realState.readMem "Mean" i = a.memory (a.base+i) := rfl

structure Output (α : Type) where
  shape : List Nat
  dtype : Storage
  device : Nat
  base : Nat
  memory : Nat → α

def makeOutput (a : Allocated c) (keepdim : Bool) (readOut : Nat → α) : Output α :=
  ⟨outputShape x.shape dims keepdim,.fp32,x.device,a.base,a.commit readOut⟩

/-- Original source launch, including arbitrary complete block order and all
physical output-exterior cells. The mathematical result uses the original
strided tensor, not the compressed source buffer as specification input. -/
theorem allocated_real_result {entry : Nat → ℝ} {c : Contiguous ℝ x dims nodup entry preexisting}
    (a : Allocated c) (tiles : List Nat)
    (schedule : tiles.Perm (List.range (gridSize (rows x.shape dims)))) :
    ∃ final, runTiles "X" "Mean" (rows x.shape dims) (columns x.shape dims) tiles a.realState = some final ∧
      (∀ row : Fin (rows x.shape dims), a.commit (final.readMem "Mean") (a.base+row.val) =
        mathematicalMean x.shape dims (x.read entry) ((batchLinear x.shape dims).symm row)) ∧
      (∀ addr, (∀ i, i < rows x.shape dims → addr ≠ a.base+i) →
        a.commit (final.readMem "Mean") addr = a.memory addr) ∧
      (∀ addr, preexisting addr → a.commit (final.readMem "Mean") addr = entry addr) := by
  obtain ⟨final,ran,values,frame⟩ := grid_result "X" "Mean" (rows x.shape dims)
    (columns x.shape dims) (by decide) tiles schedule a.realState
  have exterior (i : Nat) (hi : rows x.shape dims ≤ i) :
      final.readMem "Mean" i = a.memory (a.base+i) := by
    change final.readMem "Mean" i = a.realState.readMem "Mean" i
    simp only [BlockState.readMem,frame "Mean" i (Or.inr hi)]
  have physicalFrame := a.commit_frame (final.readMem "Mean") exterior
  refine ⟨final,ran,?_,physicalFrame,?_⟩
  · intro row
    rw [a.commit_value,values row.val row.isLt]
    unfold rowMean
    simp only [realState_input]
    have sums : (∑ col : Fin (columns x.shape dims),
        a.memory (c.base+(row.val*columns x.shape dims+col.val))) =
        ∑ col : Fin (columns x.shape dims),
          compressedValue x.shape dims nodup x.stride (x.read entry) row col := by
      apply Finset.sum_congr rfl
      intro col _
      simpa only [Nat.add_assoc] using a.input_copy row col
    rw [sums]
    exact compressed_mean_eq_mathematical x.shape dims nodup x.stride (x.read entry) row
  · intro addr hp
    rw [physicalFrame addr (by
      intro i hi equal
      apply a.fresh i hi
      simpa only [← equal] using hp)]
    exact a.protected_frame addr hp

/-- Runtime facts are only ordinary library/allocator and schedule contracts.
No output value, reduction result, kernel equivalence, or numerical law occurs
in this interface. Allocation/setup failures remain observable. -/
structure Runtime (α : Type) (x : Tensor rank) (dims : List (Fin rank)) (nodup : dims.Nodup)
    (entry : Nat → α) (preexisting : Nat → Prop) where
  compress : Option (Contiguous α x dims nodup entry preexisting)
  allocate : (c : Contiguous α x dims nodup entry preexisting) → Option (Allocated c)
  launchReady : (c : Contiguous α x dims nodup entry preexisting) → Allocated c → Bool
  order : (c : Contiguous α x dims nodup entry preexisting) → Allocated c → List Nat
  schedule : ∀ c a, (order c a).Perm (List.range (gridSize (rows x.shape dims)))

def executeReal {entry : Nat → ℝ} (rt : Runtime ℝ x dims nodup entry preexisting) (keepdim : Bool) : Option (Output ℝ) := do
  let c ← rt.compress
  let a ← rt.allocate c
  if rt.launchReady c a then
    let final ← runTiles "X" "Mean" (rows x.shape dims) (columns x.shape dims) (rt.order c a) a.realState
    return makeOutput a keepdim (final.readMem "Mean")
  else none

def HostPost (entry : Nat → ℝ) (x : Tensor rank) (dims : List (Fin rank)) (keepdim : Bool)
    (preexisting : Nat → Prop) (out : Output ℝ) : Prop :=
  out.shape = outputShape x.shape dims keepdim ∧ out.dtype = .fp32 ∧ out.device = x.device ∧
  (∀ row : Fin (rows x.shape dims), out.memory (out.base+row.val) =
    mathematicalMean x.shape dims (x.read entry) ((batchLinear x.shape dims).symm row)) ∧
  (∀ addr, preexisting addr → out.memory addr = entry addr)

/-- Successful ordinary runtime setup always reaches the actual return, and
that returned tensor has the independently specified values, metadata/frame. -/
theorem executeReal_returns {entry : Nat → ℝ} (rt : Runtime ℝ x dims nodup entry preexisting) (keepdim : Bool)
    (c : Contiguous ℝ x dims nodup entry preexisting) (a : Allocated c)
    (hc : rt.compress = some c) (ha : rt.allocate c = some a)
    (ready : rt.launchReady c a = Bool.true) :
    ∃ out, executeReal rt keepdim = some out ∧ HostPost entry x dims keepdim preexisting out := by
  obtain ⟨final,ran,values,_frame,protectedFrame⟩ := allocated_real_result a (rt.order c a) (rt.schedule c a)
  refine ⟨makeOutput a keepdim (final.readMem "Mean"),?_,rfl,rfl,rfl,values,protectedFrame⟩
  simp [executeReal,hc,ha,ready,ran]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MeanReduction.Whole
