import bench.tritonbench_g.rowwise_quantization_triton.Proofs.KernelExecution

namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.KernelResult
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false
def offset {α : Type} (p : Parameters) (s : State α) (i : TileIndex [p.width]) :=
  s.pids 0 * p.columns + i.1.val

def loaded {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (i : TileIndex [p.width]) : α :=
  M.fp32Load (if i.1.val < p.columns then (s.mem p.x (offset p s i)).read .real
    else s.undef .real p.x (offset p s i))

/-- The reduction receives zeros in inactive lanes. Undefined masked loads
are preserved by the source and have no influence on this input tile. -/
def maxInput {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (i : TileIndex [p.width]) : α :=
  if i.1.val < p.columns then M.unary (some .fp32) (.abs .real)
    (M.fp32Load ((s.mem p.x (offset p s i)).read .real))
  else M.literal (some .fp32) .real 0

def maximum {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : α :=
  M.reduceMax (some .fp32) (shape := [p.width]) ⟨0, by simp⟩ Bool.false (maxInput M p s) PUnit.unit

def scaledValue {α : Type} [Inhabited α] (M : Algebra α) (equal : α → α → Bool) (p : Parameters) (s : State α)
    (i : TileIndex [p.width]) : α :=
  M.binary (some .fp32) .real .mul (M.literal (some .fp32) .real 127)
    (M.binary (some .fp32) .real .div (M.fp32Load ((s.mem p.x (offset p s i)).read .real))
      (if equal (maximum M p s) (M.literal (some .fp32) .real 0) then
        M.literal (some .fp32) .real 1 else maximum M p s))

def maximumStored {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : α :=
  M.cast none .real .fp16 (maximum M p s)

def rowWritten {α : Type} [Inhabited α] (M : Algebra α) (round : α → BitVec 64)
    (equal : α → α → Bool) (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [p.width]).foldl (fun t i => if i.1.val < p.columns then
    t.write p.out (offset p s i) (.mk .int ((KernelExecution.signedByte (round (scaledValue M equal p s i))))) else t) s

def reference {α : Type} [Inhabited α] (M : Algebra α) (round : α → BitVec 64)
    (equal : α → α → Bool) (p : Parameters) (s : State α) : State α :=
  (rowWritten M round equal p s).write p.maxOut (s.pids 0) (.mk .fp16 (maximumStored M p s))

private theorem fold_memory {α ι : Type} (l : List ι) (r : RegionName)
    (off : ι → Nat) (value : ι → Cell α) (active : ι → Prop) [DecidablePred active] (s : State α) :
    (l.foldl (fun t i => if active i then t.write r (off i) (value i) else t) s).mem =
      l.foldl (fun mem i => fun r' o =>
        if active i ∧ r' = r ∧ o = off i then value i else mem r' o) s.mem := by
  induction l generalizing s with
  | nil => rfl
  | cons i l ih =>
    by_cases hi : active i
    · simp only [List.foldl_cons, if_pos hi, ih]
      congr 1
      funext r' o
      simp [hi, State.write]
    · simp only [List.foldl_cons, if_neg hi, ih]
      congr 1
      funext r' o
      simp [hi]

private theorem fold_expanded {α ι : Type} (l : List ι) (r : RegionName)
    (off : ι → Nat) (value : ι → Cell α) (active : ι → Prop) [DecidablePred active] (s : State α) :
    (l.foldl (fun t i => if active i then
      { t with mem := fun r' o => if r' = r ∧ o = off i then value i else t.mem r' o }
      else t) s).mem =
    l.foldl (fun mem i => fun r' o => if active i ∧ r' = r ∧ o = off i then value i else mem r' o) s.mem :=
  fold_memory l r off value active s

private theorem fold_regs {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (r : RegionName)
    (off : ι → Nat) (v : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write r (off i) (v i) else t) s).regs = s.regs := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih => simp only [List.foldl_cons, ih]; split <;> rfl

private theorem fold_pids {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (r : RegionName)
    (off : ι → Nat) (v : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write r (off i) (v i) else t) s).pids = s.pids := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih => simp only [List.foldl_cons, ih]; split <;> rfl

theorem source_run {α : Type} [Inhabited α] (M : Algebra α) (round : α → BitVec 64)
    (equal : α → α → Bool) (p : Parameters) (s : State α)
    (hwidth : 0 < p.width) :
    ∃ t, KernelExecution.exec (backend M equal) round (kernel p) s = some t ∧ t.mem = (reference M round equal p s).mem := by
  simp +contextual (config := { maxSteps := 1000000 }) [kernel, RowwiseQuantizationTriton.source, ComputeKernel.surfaceBody,
    KernelExecution.exec, KernelExecution.run, KernelExecution.step, KernelExecution.eval,
    evalExpr, evalComputeOp, evalOp_unfold, store, backend, numericEq,
    State.setReg, reference, rowWritten, scaledValue, maximumStored, maximum, maxInput, loaded, offset,
    ofFloat, toFloat, Region.cast, ComputeDType.eraseDType, bop, numeric,
    numericLt, natLt, Option.bind, fold_regs, fold_pids, TileShape.axisDim, hwidth]
  try simp only [fold_memory]
  simp +contextual (config := { maxSteps := 1000000 }) [State.write, fold_memory, fold_expanded, maxInput, show (127.0 : ℝ) = 127 by norm_num]
  have hmax : (fun (i : TileIndex [p.width]) =>
      if i.1.val < p.columns then M.unary (some .fp32) (.abs .real)
        (M.fp32Load (if i.1.val < p.columns then
          (s.mem p.x (s.pids 0 * p.columns + i.1.val)).read .real
          else s.undef .real p.x (s.pids 0 * p.columns + i.1.val)))
      else M.literal (some .fp32) .real 0.0) = maxInput M p s := by
    funext i
    by_cases hi : i.1.val < p.columns <;> simp [hi, maxInput, offset, show (0.0 : ℝ) = 0 by norm_num]
  simp only [hmax]
  funext r o
  split
  · rfl
  · apply congrArg (fun (mem : RegionName → Nat → Cell α) => mem r o)
    congr 1
    funext mem i r2 o2
    split
    · rename_i h
      simp [h.1, show (0.0 : ℝ) = 0 by norm_num, show (1.0 : ℝ) = 1 by norm_num]
      exact signedBits_int_mod 8 _
    · rfl

theorem offset_injective {α : Type} (p : Parameters) (s : State α) : Function.Injective (offset p s) := by
  intro i j h
  have hf : i.1 = j.1 := Fin.ext (Nat.add_left_cancel h)
  exact Prod.ext hf (Subsingleton.elim _ _)

def Outside {α : Type} (p : Parameters) (s : State α) (r : RegionName) (o : Nat) : Prop :=
  (r ≠ p.maxOut ∨ o ≠ s.pids 0) ∧
    (r ≠ p.out ∨ ∀ i : TileIndex [p.width], i.1.val < p.columns → o ≠ offset p s i)

theorem source_correct {α : Type} [Inhabited α] (M : Algebra α) (round : α → BitVec 64)
    (equal : α → α → Bool) (p : Parameters) (s : State α)
    (hwidth : 0 < p.width) (hregions : p.out ≠ p.maxOut) :
    ∃ t, KernelExecution.exec (backend M equal) round (kernel p) s = some t ∧
      (∀ i : TileIndex [p.width], i.1.val < p.columns →
        t.mem p.out (offset p s i) = .mk .int ((KernelExecution.signedByte (round (scaledValue M equal p s i))))) ∧
      t.mem p.maxOut (s.pids 0) = .mk .fp16 (maximumStored M p s) ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hm⟩ := source_run M round equal p s hwidth
  refine ⟨t, he, ?_, ?_, ?_⟩
  · intro i hi
    rw [hm]
    simp only [reference, State.write, hregions, false_and, if_false]
    exact State.masked_scatter_readback p.out (offset p s) _ (fun i => i.1.val < p.columns)
      s i hi (offset_injective p s)
  · rw [hm]
    simp [reference, State.write]
  · intro r o h
    rw [hm]
    have hnot : ¬ (r = p.maxOut ∧ o = s.pids 0) := by
      rcases h.1 with hr | ho
      · exact fun hx => hr hx.1
      · exact fun hx => ho hx.2
    simp only [reference, State.write, hnot, if_false]
    unfold rowWritten
    apply State.masked_scatter_frame
    rcases h.2 with hr | ho
    · exact Or.inl hr
    · exact Or.inr (fun i _ hi => ho i hi)
#axiomsClean source_run
#axiomsClean source_correct
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.KernelResult
