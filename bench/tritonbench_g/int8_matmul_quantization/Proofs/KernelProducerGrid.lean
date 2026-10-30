import bench.tritonbench_g.int8_matmul_quantization.Proofs.KernelProducerReadback
import bench.tritonbench_g.rowwise_quantization_triton.Proofs.Memory

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducerGrid
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

structure Geometry where
  rows : Nat
  columns : Nat
  width : Nat
  positive : 0 < width

structure Schedule (p : Geometry) where
  order : List (Fin p.rows)
  complete : order.Perm (List.finRange p.rows)

def parameters (p : Geometry) : KernelProducer.Parameters :=
  ⟨"x", "q", "scales", p.rows, p.columns, p.width, p.columns, 1, p.columns, 1, 1⟩
def atRow (s : State α) (r : Nat) : State α :=
  { s with pids := Function.update s.pids 0 r }
def advance (M : Algebra α) (integer8 : α → BitVec 8)
    (p : Geometry) (s : State α) (r : Fin p.rows) : State α :=
  (KernelProducer.execution M integer8 (parameters p) (atRow s r) p.positive).val
def finalState (M : Algebra α) (integer8 : α → BitVec 8)
    (p : Geometry) : List (Fin p.rows) → State α → State α
  | [], s => s
  | r :: tail, s => finalState M integer8 p tail (advance M integer8 p s r)

/-- Launches the complete canonical quotation for each scheduled source row.
An empty row grid performs no invocation and imposes no unused K bound. -/
def run (M : Algebra α) (integer8 : α → BitVec 8)
    (p : Geometry) : List (Fin p.rows) → State α → Option (State α)
  | [], s => some s
  | r :: tail, s => do
      let next ← FP.Structural.exec (KernelProducer.backend M integer8)
        (KernelSource.quantize (KernelProducer.parameters (parameters p))) (atRow s r)
      run M integer8 p tail next

theorem run_final (M : Algebra α) (integer8 : α → BitVec 8)
    (p : Geometry) (order : List (Fin p.rows)) (s : State α) :
    run M integer8 p order s = some (finalState M integer8 p order s) := by
  induction order generalizing s with
  | nil => rfl
  | cons r tail ih =>
    have hs := (KernelProducer.execution M integer8 (parameters p) (atRow s r) p.positive).property
    change FP.Structural.exec (KernelProducer.backend M integer8)
      (KernelSource.quantize (KernelProducer.parameters (parameters p))) (atRow s r) =
      some (advance M integer8 p s r) at hs
    simpa only [run, finalState, hs, Option.bind_some] using ih (advance M integer8 p s r)

def scaled (M : Algebra α) (p : Geometry) (s : State α) (r : Nat) (c : Fin p.width) : α :=
  KernelProducer.quotient M (parameters p) (atRow s r) (0,c,PUnit.unit)
def scale (M : Algebra α) (p : Geometry) (s : State α) (r : Nat) : α :=
  M.cast none .real .fp16 (KernelProducer.scale M (parameters p) (atRow s r))

theorem advance_correct (M : Algebra α) (integer8 : α → BitVec 8)
    (p : Geometry) (s : State α) (r : Fin p.rows) :
    (∀ c : Fin p.width, c.val < p.columns →
      (advance M integer8 p s r).mem "q" (r.val*p.columns+c.val) =
        .mk .int (integer8 (scaled M p s r c)).toInt) ∧
    (advance M integer8 p s r).mem "scales" r = .mk .fp16 (scale M p s r) ∧
    (∀ region offset, KernelProducer.Outside (parameters p) (atRow s r) region offset →
      (advance M integer8 p s r).mem region offset = s.mem region offset) := by
  obtain ⟨t, he, hq, hs, hf⟩ := KernelProducer.source_correct M integer8 (parameters p)
    (atRow s r) p.positive rfl (by simp [parameters])
  have hex := (KernelProducer.execution M integer8 (parameters p) (atRow s r) p.positive).property
  have ht : advance M integer8 p s r = t := Option.some.inj (hex.symm.trans he)
  rw [ht]
  refine ⟨?_, ?_, hf⟩
  · intro c hc
    simpa [scaled, KernelProducer.outputOffset, parameters, atRow, Nat.mod_eq_of_lt r.isLt]
      using hq (0,c,PUnit.unit) hc
  · simpa [scale, parameters, atRow] using hs

def SameInput (initial current : State α) : Prop := ∀ i, current.mem "x" i = initial.mem "x" i
theorem advance_input (M : Algebra α) (integer8 : α → BitVec 8)
    (p : Geometry) (s : State α) (r : Fin p.rows) : SameInput s (advance M integer8 p s r) := by
  intro i
  apply (advance_correct M integer8 p s r).2.2
  exact ⟨Or.inl (by simp [parameters]), Or.inl (by simp [parameters])⟩

theorem scale_same (M : Algebra α) (p : Geometry) (initial current : State α)
    (h : SameInput initial current) (r : Nat) : scale M p current r = scale M p initial r := by
  unfold SameInput at h
  simp [scale, KernelProducer.scale, KernelProducer.maximum, KernelProducer.loaded,
    KernelProducer.inputOffset, parameters, atRow, h]
theorem scaled_same (M : Algebra α) (p : Geometry) (initial current : State α)
    (h : SameInput initial current) (r : Nat) (c : Fin p.width) :
    scaled M p current r c = scaled M p initial r c := by
  unfold SameInput at h
  simp [scaled, KernelProducer.quotient, KernelProducer.scale, KernelProducer.maximum,
    KernelProducer.loaded, KernelProducer.inputOffset, parameters, atRow, h]

theorem final_frame_order (M : Algebra α) (integer8 : α → BitVec 8)
    (p : Geometry) (order : List (Fin p.rows)) (s : State α) (region : RegionName) (offset : Nat)
    (miss : ∀ r ∈ order, KernelProducer.Outside (parameters p) (atRow s r) region offset) :
    (finalState M integer8 p order s).mem region offset = s.mem region offset := by
  induction order generalizing s with
  | nil => rfl
  | cons r tail ih =>
    simp only [finalState]
    rw [ih]
    · exact (advance_correct M integer8 p s r).2.2 region offset (miss r (by simp))
    · intro q hq
      simpa [KernelProducer.Outside, KernelProducer.outputOffset, atRow] using miss q (by simp [hq])

theorem final_quantized (M : Algebra α) (integer8 : α → BitVec 8)
    (p : Geometry) (order : List (Fin p.rows)) (s : State α) (r : Fin p.rows) (hr : r ∈ order)
    (c : Fin p.width) (hc : c.val < p.columns) :
    (finalState M integer8 p order s).mem "q" (r.val*p.columns+c.val) =
      .mk .int (integer8 (scaled M p s r c)).toInt := by
  induction order generalizing s with
  | nil => simp at hr
  | cons q tail ih =>
    by_cases ht : r ∈ tail
    · simp only [finalState]
      rw [ih _ ht, scaled_same M p s _ (advance_input M integer8 p s q) r c]
    · have he : r = q := by simpa [ht] using hr
      subst q
      simp only [finalState]
      rw [final_frame_order M integer8 p tail (advance M integer8 p s r)]
      · exact (advance_correct M integer8 p s r).1 c hc
      · intro q hq
        refine ⟨Or.inl (by simp [parameters]), Or.inr ?_⟩
        intro j hj heq
        have roweq := (RowwiseQuantizationTriton.linear_injective p.columns r q c.val j.2.1.val hc hj
          (by simpa [KernelProducer.outputOffset, parameters, atRow, Nat.mod_eq_of_lt q.isLt] using heq)).1
        exact ht ((Fin.ext roweq).symm ▸ hq)

theorem final_scale (M : Algebra α) (integer8 : α → BitVec 8)
    (p : Geometry) (order : List (Fin p.rows)) (s : State α) (r : Fin p.rows) (hr : r ∈ order) :
    (finalState M integer8 p order s).mem "scales" r = .mk .fp16 (scale M p s r) := by
  induction order generalizing s with
  | nil => simp at hr
  | cons q tail ih =>
    by_cases ht : r ∈ tail
    · simp only [finalState]
      rw [ih _ ht, scale_same M p s _ (advance_input M integer8 p s q) r]
    · have he : r = q := by simpa [ht] using hr
      subst q
      simp only [finalState]
      rw [final_frame_order M integer8 p tail (advance M integer8 p s r)]
      · exact (advance_correct M integer8 p s r).2.1
      · intro q hq
        refine ⟨Or.inr ?_, Or.inl (by simp [parameters])⟩
        simp only [parameters, atRow, Function.update_self, Nat.mul_one]
        exact fun he => ht ((Fin.ext he).symm ▸ hq)

theorem final_frame (M : Algebra α) (integer8 : α → BitVec 8)
    (p : Geometry) (order : List (Fin p.rows)) (s : State α) (region : RegionName) (offset : Nat)
    (hq : region ≠ "q" ∨ p.rows*p.columns ≤ offset)
    (hs : region ≠ "scales" ∨ p.rows ≤ offset) :
    (finalState M integer8 p order s).mem region offset = s.mem region offset := by
  apply final_frame_order
  intro r _
  constructor
  · rcases hs with hs | hs
    · exact Or.inl hs
    · right
      simp only [parameters, atRow, Function.update_self, Nat.mul_one]
      have := r.isLt
      omega
  · rcases hq with hq | hq
    · exact Or.inl hq
    · right
      intro i hi he
      have bound := RowwiseQuantizationTriton.linear_bound p.rows p.columns r i.2.1.val r.isLt hi
      simp only [KernelProducer.outputOffset, parameters, atRow, Function.update_self,
        Nat.mod_eq_of_lt r.isLt, Nat.mul_one] at he
      omega

/-- Complete actual producer grid, all active data and scale outputs, and
all exterior logical memory. Numerical interpretation is a later scalar
specialization, not a premise about any output or source execution. -/
theorem source_correct (M : Algebra α) (integer8 : α → BitVec 8)
    (p : Geometry) (schedule : Schedule p) (initial : State α) :
    ∃ final, run M integer8 p schedule.order initial = some final ∧
      (∀ r : Fin p.rows, ∀ c : Fin p.width, c.val < p.columns →
        final.mem "q" (r.val*p.columns+c.val) = .mk .int (integer8 (scaled M p initial r c)).toInt) ∧
      (∀ r : Fin p.rows, final.mem "scales" r = .mk .fp16 (scale M p initial r)) ∧
      (∀ (region : RegionName) offset, (region ≠ "q" ∨ p.rows*p.columns ≤ offset) →
        (region ≠ "scales" ∨ p.rows ≤ offset) → final.mem region offset = initial.mem region offset) := by
  refine ⟨finalState M integer8 p schedule.order initial,
    run_final M integer8 p schedule.order initial, ?_, ?_, ?_⟩
  · intro r c hc
    exact final_quantized M integer8 p schedule.order initial r
      (schedule.complete.mem_iff.mpr (List.mem_finRange r)) c hc
  · intro r
    exact final_scale M integer8 p schedule.order initial r
      (schedule.complete.mem_iff.mpr (List.mem_finRange r))
  · exact final_frame M integer8 p schedule.order initial

#axiomsClean run_final
#axiomsClean final_quantized
#axiomsClean final_scale
#axiomsClean final_frame_order
#axiomsClean source_correct
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducerGrid
