import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.KernelResult
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.Memory

namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.KernelGrid
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section
set_option maxHeartbeats 2500000

variable {α : Type} [Inhabited α]

def parameters (p : Launch) : Parameters := ⟨"x", "q", "m", p.columns, p.width⟩
def atRow (s : State α) (r : Nat) : State α :=
  { s with pids := Function.update s.pids 0 r }

def advance (M : Algebra α) (round : α → BitVec 64) (equal : α → α → Bool)
    (p : Launch) (s : State α) (r : Nat) : State α :=
  (execution M round equal (parameters p) (atRow s r)
    (lt_of_lt_of_le p.positive p.covers)).val

def finalState (M : Algebra α) (round : α → BitVec 64) (equal : α → α → Bool)
    (p : Launch) : List Nat → State α → State α
  | [], s => s
  | r :: tail, s => finalState M round equal p tail (advance M round equal p s r)

/-- The actual selected source, using the documented local llrint/register
extension, is executed once per scheduled row. -/
def run (M : Algebra α) (round : α → BitVec 64) (equal : α → α → Bool)
    (p : Launch) : List Nat → State α → Option (State α)
  | [], s => some s
  | r :: tail, s => do
      let next ← KernelExecution.exec (backend M equal) round (kernel (parameters p)) (atRow s r)
      run M round equal p tail next

theorem run_final (M : Algebra α) (round : α → BitVec 64) (equal : α → α → Bool)
    (p : Launch) (order : List Nat) (s : State α) :
    run M round equal p order s = some (finalState M round equal p order s) := by
  induction order generalizing s with
  | nil => rfl
  | cons r tail ih =>
    have hs := (execution M round equal (parameters p) (atRow s r)
      (lt_of_lt_of_le p.positive p.covers)).property
    change KernelExecution.exec (backend M equal) round (kernel (parameters p)) (atRow s r) =
      some (advance M round equal p s r) at hs
    simpa only [run, finalState, hs, Option.bind_some] using ih (advance M round equal p s r)

theorem advance_correct (M : Algebra α) (round : α → BitVec 64) (equal : α → α → Bool)
    (p : Launch) (s : State α) (r : Nat) :
    (∀ i : TileIndex [p.width], i.1.val < p.columns →
      (advance M round equal p s r).mem "q" (r*p.columns+i.1.val) =
        .mk .int ((KernelExecution.signedByte (round (KernelResult.scaledValue M equal (parameters p) (atRow s r) i))))) ∧
    (advance M round equal p s r).mem "m" r =
      .mk .fp16 (KernelResult.maximumStored M (parameters p) (atRow s r)) ∧
    (∀ region offset, KernelResult.Outside (parameters p) (atRow s r) region offset →
      (advance M round equal p s r).mem region offset = s.mem region offset) := by
  obtain ⟨t, he, hq, hm, hf⟩ := KernelResult.source_correct M round equal (parameters p)
    (atRow s r) (lt_of_lt_of_le p.positive p.covers) (by simp [parameters])
  have hs := (execution M round equal (parameters p) (atRow s r)
    (lt_of_lt_of_le p.positive p.covers)).property
  change KernelExecution.exec (backend M equal) round (kernel (parameters p)) (atRow s r) =
    some (advance M round equal p s r) at hs
  have ht : advance M round equal p s r = t := Option.some.inj (hs.symm.trans he)
  rw [ht]
  simpa [KernelResult.offset, parameters, atRow] using And.intro hq (And.intro hm hf)

def SameInput (initial current : State α) : Prop := ∀ i, current.mem "x" i = initial.mem "x" i

theorem advance_input (M : Algebra α) (round : α → BitVec 64) (equal : α → α → Bool)
    (p : Launch) (s : State α) (r : Nat) : SameInput s (advance M round equal p s r) := by
  intro i
  apply (advance_correct M round equal p s r).2.2
  exact ⟨Or.inl (by simp [parameters]), Or.inl (by simp [parameters])⟩

theorem final_input (M : Algebra α) (round : α → BitVec 64) (equal : α → α → Bool)
    (p : Launch) (order : List Nat) (s : State α) : SameInput s (finalState M round equal p order s) := by
  induction order generalizing s with
  | nil => exact fun _ => rfl
  | cons r tail ih =>
    intro i
    exact (ih (advance M round equal p s r) i).trans (advance_input M round equal p s r i)

theorem maximum_same (M : Algebra α) (p : Launch) (initial s : State α)
    (hs : SameInput initial s) (r : Nat) :
    KernelResult.maximum M (parameters p) (atRow s r) =
      KernelResult.maximum M (parameters p) (atRow initial r) := by
  unfold SameInput at hs
  unfold KernelResult.maximum
  congr 1
  funext i
  simp [KernelResult.maxInput, KernelResult.offset, parameters, atRow, hs]

theorem value_same (M : Algebra α) (equal : α → α → Bool) (p : Launch) (initial s : State α)
    (hs : SameInput initial s) (r : Nat) (i : TileIndex [p.width]) :
    KernelResult.scaledValue M equal (parameters p) (atRow s r) i =
      KernelResult.scaledValue M equal (parameters p) (atRow initial r) i := by
  have hm := maximum_same M p initial s hs r
  unfold SameInput at hs
  simp only [KernelResult.scaledValue, hm]
  simp [KernelResult.offset, parameters, atRow, hs]

theorem final_frame_order (M : Algebra α) (round : α → BitVec 64) (equal : α → α → Bool)
    (p : Launch) (order : List Nat) (s : State α) (region : RegionName) (offset : Nat)
    (miss : ∀ r ∈ order, KernelResult.Outside (parameters p) (atRow s r) region offset) :
    (finalState M round equal p order s).mem region offset = s.mem region offset := by
  induction order generalizing s with
  | nil => rfl
  | cons r tail ih =>
    simp only [finalState]
    rw [ih]
    · exact (advance_correct M round equal p s r).2.2 region offset (miss r (by simp))
    · intro q hq
      simpa [KernelResult.Outside, KernelResult.offset, atRow] using miss q (by simp [hq])

theorem final_quantized (M : Algebra α) (round : α → BitVec 64) (equal : α → α → Bool)
    (p : Launch) (order : List Nat) (s : State α) (r : Nat) (hr : r ∈ order)
    (i : TileIndex [p.width]) (hi : i.1.val < p.columns) :
    (finalState M round equal p order s).mem "q" (r*p.columns+i.1.val) =
      .mk .int ((KernelExecution.signedByte (round (KernelResult.scaledValue M equal (parameters p) (atRow s r) i)))) := by
  induction order generalizing s with
  | nil => simp at hr
  | cons q tail ih =>
    by_cases ht : r ∈ tail
    · simp only [finalState]
      rw [ih _ ht]
      rw [value_same M equal p s _ (advance_input M round equal p s q) r i]
    · have he : r = q := by simpa [ht] using hr
      subst q
      simp only [finalState]
      rw [final_frame_order M round equal p tail (advance M round equal p s r)]
      · exact (advance_correct M round equal p s r).1 i hi
      · intro q hq
        refine ⟨Or.inl (by simp [parameters]), Or.inr ?_⟩
        intro j hj heq
        have hrow := (linear_injective p.columns r q i.1.val j.1.val hi hj
          (by simpa [KernelResult.offset, parameters, atRow] using heq)).1
        exact ht (hrow.symm ▸ hq)

theorem final_maximum (M : Algebra α) (round : α → BitVec 64) (equal : α → α → Bool)
    (p : Launch) (order : List Nat) (s : State α) (r : Nat) (hr : r ∈ order) :
    (finalState M round equal p order s).mem "m" r =
      .mk .fp16 (KernelResult.maximumStored M (parameters p) (atRow s r)) := by
  induction order generalizing s with
  | nil => simp at hr
  | cons q tail ih =>
    by_cases ht : r ∈ tail
    · simp only [finalState]
      rw [ih _ ht]
      simp only [KernelResult.maximumStored,
        maximum_same M p s _ (advance_input M round equal p s q) r]
    · have he : r = q := by simpa [ht] using hr
      subst q
      simp only [finalState]
      rw [final_frame_order M round equal p tail (advance M round equal p s r)]
      · exact (advance_correct M round equal p s r).2.1
      · intro q hq
        refine ⟨Or.inr ?_, Or.inl (by simp [parameters])⟩
        simpa [atRow] using (fun he : r = q => ht (he.symm ▸ hq))

#axiomsClean run_final
#axiomsClean final_quantized
#axiomsClean final_maximum
#axiomsClean final_frame_order
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.KernelGrid
