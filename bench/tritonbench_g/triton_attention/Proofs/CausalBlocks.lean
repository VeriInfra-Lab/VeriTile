import bench.tritonbench_g.triton_attention.Sources.Geometry

namespace VeriTile.Bench.Repairs.TritonAttention.Whole001
namespace Math

variable {N D : Nat}

/-- The visible keys in the actual source block, after its causal mask. -/
def visibleBlock (i : Fin N) (b : Nat) : Finset (Fin N) :=
  Finset.univ.filter (fun j => j ≤ i ∧ j.val / 128 = b)

/-- The visible keys after k source iterations. -/
def visiblePrefix (i : Fin N) (k : Nat) : Finset (Fin N) :=
  Finset.univ.filter (fun j => j ≤ i ∧ j.val / 128 < k)

theorem visibleBlock_nonempty (i : Fin N) (b : Nat) (hb : b ≤ i.val / 128) :
    (visibleBlock i b).Nonempty := by
  have hbi : b * 128 ≤ i.val := by omega
  let j : Fin N := ⟨b * 128,lt_of_le_of_lt hbi i.isLt⟩
  refine ⟨j,?_⟩
  simp only [visibleBlock,Finset.mem_filter,Finset.mem_univ,true_and]
  constructor
  · exact hbi
  · simp [j]

theorem visiblePrefix_nonempty (i : Fin N) (k : Nat) (hk : 0 < k) :
    (visiblePrefix i k).Nonempty := by
  let j : Fin N := ⟨0,by have := i.isLt; omega⟩
  refine ⟨j,?_⟩
  simp only [visiblePrefix,Finset.mem_filter,Finset.mem_univ,true_and]
  exact ⟨Nat.zero_le _,by simpa only [j,Nat.zero_div] using hk⟩

theorem visiblePrefix_one (i : Fin N) : visiblePrefix i 1 = visibleBlock i 0 := by
  ext j
  simp only [visiblePrefix,visibleBlock,Finset.mem_filter,Finset.mem_univ,true_and]
  omega

theorem visiblePrefix_step (i : Fin N) (k : Nat) :
    visiblePrefix i (k + 1) = visiblePrefix i k ∪ visibleBlock i k := by
  ext j
  simp only [visiblePrefix,visibleBlock,Finset.mem_filter,Finset.mem_univ,
    true_and,Finset.mem_union]
  omega

theorem visiblePrefix_disjoint (i : Fin N) (k : Nat) :
    Disjoint (visiblePrefix i k) (visibleBlock i k) := by
  apply Finset.disjoint_left.mpr
  intro j hp hb
  simp only [visiblePrefix,visibleBlock,Finset.mem_filter,Finset.mem_univ,true_and] at hp hb
  omega

theorem visiblePrefix_final (i : Fin N) :
    visiblePrefix i (i.val / 128 + 1) = causalSet i := by
  ext j
  simp only [visiblePrefix,causalSet,Finset.mem_filter,Finset.mem_univ,true_and]
  constructor
  · exact fun h => h.1
  · intro h
    exact ⟨h,by have : j.val ≤ i.val := h; omega⟩

/-- The erased-real source row recurrence with the actual increasing
128-key schedule, starting from its first iteration. The proof never receives
a caller-supplied partition, maximum, normalizer, or output equation. -/
noncomputable def sourceForwardRow (Q K V : Tensor N D) (scale : ℝ) (i : Fin N) :
    Nat → OnlineRow D
  | 0 => onlineFirst (score Q K scale i) V (visibleBlock i 0)
      (visibleBlock_nonempty i 0 (Nat.zero_le _))
  | t + 1 => if h : t + 1 ≤ i.val / 128 then
      onlineStep (score Q K scale i) V (sourceForwardRow Q K V scale i t)
        (visibleBlock i (t + 1)) (visibleBlock_nonempty i (t + 1) h)
    else sourceForwardRow Q K V scale i t

theorem sourceForwardRow_invariant (Q K V : Tensor N D) (scale : ℝ) (i : Fin N)
    (k : Nat) (hk : k ≤ i.val / 128) :
    sourceForwardRow Q K V scale i k =
      batchRow (score Q K scale i) V (visiblePrefix i (k + 1))
        (visiblePrefix_nonempty i (k + 1) (by omega)) := by
  induction k with
  | zero =>
    rw [sourceForwardRow,onlineFirst_eq_batchRow]
    simp only [Nat.zero_add,visiblePrefix_one]
  | succ k ih =>
    have hprev : k ≤ i.val / 128 := by omega
    rw [sourceForwardRow,dif_pos hk,ih hprev]
    rw [onlineStep_eq_batchRow _ _ _ _ _ _ (visiblePrefix_disjoint i (k + 1))]
    simp only [visiblePrefix_step]

/-- The entire source forward loop produces the three saved tensors derived
from Q/K/V, for every query row. This is an erased-real interpretation;
it is not an assertion that machine FP32/FP16 operations are exact. -/
theorem sourceForwardRow_eq_saved (Q K V : Tensor N D) (scale : ℝ) (i : Fin N) :
    sourceForwardRow Q K V scale i (i.val / 128) =
      ⟨savedMax Q K scale i,savedNormalizer Q K scale i,savedOutput Q K V scale i⟩ := by
  rw [sourceForwardRow_invariant _ _ _ _ _ _ (le_refl _)]
  simpa only [visiblePrefix_final] using causal_batchRow Q K V scale i

noncomputable def sourceForwardOutput (Q K V : Tensor N D) (scale : ℝ) : Tensor N D :=
  fun i => (sourceForwardRow Q K V scale i (i.val / 128)).output
noncomputable def sourceForwardL (Q K V : Tensor N D) (scale : ℝ) : Fin N → ℝ :=
  fun i => (sourceForwardRow Q K V scale i (i.val / 128)).normalizer
noncomputable def sourceForwardM (Q K V : Tensor N D) (scale : ℝ) : Fin N → ℝ :=
  fun i => (sourceForwardRow Q K V scale i (i.val / 128)).maximum

theorem sourceForward_tensors (Q K V : Tensor N D) (scale : ℝ) :
    sourceForwardOutput Q K V scale = savedOutput Q K V scale ∧
    sourceForwardL Q K V scale = savedNormalizer Q K scale ∧
    sourceForwardM Q K V scale = savedMax Q K scale := by
  constructor
  · funext i
    simp [sourceForwardOutput,sourceForwardRow_eq_saved]
  constructor <;> funext i
  · simp [sourceForwardL,sourceForwardRow_eq_saved]
  · simp [sourceForwardM,sourceForwardRow_eq_saved]

#axiomsClean visibleBlock_nonempty
#axiomsClean visiblePrefix_final
#axiomsClean sourceForwardRow_invariant
#axiomsClean sourceForwardRow_eq_saved
#axiomsClean sourceForward_tensors
#specNonCircular attention avoiding [sourceForwardRow,sourceForwardOutput]
end Math
end VeriTile.Bench.Repairs.TritonAttention.Whole001
