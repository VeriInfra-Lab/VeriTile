import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Events

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

/-- A scalar binary-max network, with explicit masked zero leaves. No result
of a reduction is accepted as a numerical primitive. -/
inductive Reduction (K : Nat) where
  | zero
  | lane (k : Fin K)
  | maximum (left right : Reduction K)
def Reduction.terms : Reduction K → List (Fin K)
  | .zero => []
  | .lane k => [k]
  | .maximum left right => left.terms++right.terms
structure ReductionPlan (K : Nat) where
  tree : Reduction K
  complete : tree.terms.Perm (List.finRange K)

def Reduction.eval (a : Ops I F H) (value : Fin K → F) : Reduction K → F
  | .zero => a.fzero
  | .lane k => a.absolute (value k)
  | .maximum left right => a.maximum (left.eval a value) (right.eval a value)
def Reduction.code (input : Fin K → Address) : Reduction K → Code I F H F
  | .zero => emit .fzero
  | .lane k => do
      let value ← emit (.read32 (input k))
      emit (.absolute value)
  | .maximum left right => do
      let x ← left.code input
      let y ← right.code input
      emit (.maximum x y)
theorem Reduction.eval_code (a : Ops I F H) (input : Fin K → Address) (tree : Reduction K)
    (m : Memory F H) :
    (tree.code input).eval a m=(tree.eval a (fun k => read32 a.toOps m (input k)),m) := by
  induction tree with
  | zero => rfl
  | lane k => rfl
  | maximum left right hl hr =>
    simp only [Reduction.code,bind,eval_bind,hl,hr,Prod.fst,Prod.snd,eval_emit,Event.eval,Reduction.eval]

/-- Independent canonical mathematical maximum; no execution, tree or
compiler selection appears in this target. -/
def maximumAbs (value : Nat → ℝ) : Nat → ℝ
  | 0 => 0
  | n+1 => max (maximumAbs value n) |value n|
theorem maximumAbs_nonnegative (value : Nat → ℝ) (n : Nat) : 0≤maximumAbs value n := by
  induction n with
  | zero => rfl
  | succ n ih => exact le_trans ih (le_max_left _ _)
theorem maximumAbs_covers (value : Nat → ℝ) (n k : Nat) (hk : k<n) : |value k|≤maximumAbs value n := by
  induction n with
  | zero => omega
  | succ n ih =>
    by_cases h : k<n
    · exact le_trans (ih h) (le_max_left _ _)
    · have eq : k=n := by omega
      rw [eq]
      exact le_max_right _ _
theorem maximumAbs_le (value : Nat → ℝ) (n : Nat) (upper : ℝ) (positive : 0≤upper)
    (bound : ∀ k,k<n → |value k|≤upper) : maximumAbs value n≤upper := by
  induction n with
  | zero => exact positive
  | succ n ih => exact max_le (ih (fun k hk => bound k (by omega))) (bound n (by omega))

def Reduction.real (value : Fin K → ℝ) : Reduction K → ℝ
  | .zero => 0
  | .lane k => |value k|
  | .maximum left right => max (left.real value) (right.real value)
theorem Reduction.real_nonnegative (value : Fin K → ℝ) (tree : Reduction K) : 0≤tree.real value := by
  induction tree with
  | zero => rfl
  | lane k => exact abs_nonneg _
  | maximum left right hl hr => exact le_trans hl (le_max_left _ _)
theorem Reduction.real_covers (value : Fin K → ℝ) (tree : Reduction K) (k : Fin K)
    (present : k∈tree.terms) : |value k|≤tree.real value := by
  induction tree with
  | zero => simp [terms] at present
  | lane j =>
    have eq : k=j := by simpa [terms] using present
    simp [Reduction.real,eq]
  | maximum left right hl hr =>
    rcases List.mem_append.mp present with h|h
    · exact le_trans (hl h) (le_max_left _ _)
    · exact le_trans (hr h) (le_max_right _ _)
theorem Reduction.real_le (value : Fin K → ℝ) (tree : Reduction K) (upper : ℝ) (positive : 0≤upper)
    (bound : ∀ k,k∈tree.terms → |value k|≤upper) : tree.real value≤upper := by
  induction tree with
  | zero => exact positive
  | lane k => exact bound k (by simp [terms])
  | maximum left right hl hr =>
    exact max_le (hl (fun k h => bound k (List.mem_append_left _ h)))
      (hr (fun k h => bound k (List.mem_append_right _ h)))

theorem ReductionPlan.real_target (plan : ReductionPlan K) (value : Nat → ℝ) :
    plan.tree.real (fun k => value k)=maximumAbs value K := by
  apply le_antisymm
  · exact plan.tree.real_le _ _ (maximumAbs_nonnegative value K)
      (fun k _ => maximumAbs_covers value K k k.isLt)
  · apply maximumAbs_le value K _ (plan.tree.real_nonnegative _)
    intro k hk
    exact plan.tree.real_covers (fun j : Fin K => value j.val) ⟨k,hk⟩ ((plan.complete.mem_iff).mpr (by simp))

#specNonCircular maximumAbs avoiding [Reduction.eval,Reduction.code]
#axiomsClean Reduction.eval_code
#axiomsClean ReductionPlan.real_target
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization
