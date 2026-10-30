import bench.tritonbench_g.ksoftmax_triton.Proofs.KernelExecution

namespace VeriTile.Bench.Repairs.Ksoftmax.KernelSchedule
open VeriTile.Triton FP.Structural
noncomputable section

theorem coordinate (B Q K D offset : Nat) (covered : K≤D) (inside : offset<B*Q*K) :
    ∃ (b : Fin B) (r : Fin Q) (i : Fin D), i.val<K ∧ Native.flat Q K b r i=offset := by
  have hk : 0<K := by
    by_contra h
    have hz : K=0 := by omega
    simp [hz] at inside
  have hq : 0<Q := by
    by_contra h
    have hz : Q=0 := by omega
    simp [hz] at inside
  have rowBound : offset/K<B*Q := (Nat.div_lt_iff_lt_mul hk).mpr inside
  have batchBound : offset/K/Q<B := (Nat.div_lt_iff_lt_mul hq).mpr rowBound
  have laneBound : offset%K<K := Nat.mod_lt offset hk
  refine ⟨⟨offset/K/Q,batchBound⟩,⟨offset/K%Q,Nat.mod_lt _ hq⟩,
    ⟨offset%K,lt_of_lt_of_le laneBound covered⟩,laneBound,?_⟩
  have r := Nat.mod_add_div (offset/K) Q
  have k := Nat.mod_add_div offset K
  dsimp [Native.flat]
  calc
    offset/K/Q*(Q*K)+(offset/K%Q)*K+offset%K=
        (offset/K%Q+Q*(offset/K/Q))*K+offset%K := by ring
    _=(offset/K)*K+offset%K := by rw [r]
    _=offset := by simpa [Nat.mul_comm,Nat.add_comm] using k

/-- All complete row schedules have identical entire source memory. Every row
still runs the complete source; only the order of independent programs changes.
This includes all exceptional words and every source algebra. -/
theorem memory_eq {α : Type} [Inhabited α] (model : Algebra α) (op : Operation)
    (B Q K D : Nat) (hD : 0<D) (covered : K≤D)
    (schedule : List (Fin B × Fin Q)) (complete : ∀ b r,(b,r)∈schedule) (s : State α) :
    (runRows model op "dst" "left" "right" B Q K D hD schedule s).mem=
      (runRows model op "dst" "left" "right" B Q K D hD (programs B Q) s).mem := by
  funext region offset
  by_cases same : region="dst"
  · subst region
    by_cases inside : offset<B*Q*K
    · obtain ⟨b,r,i,hi,eq⟩ := coordinate B Q K D offset covered inside
      rw [←eq,rows_observe model op "dst" "left" "right" B Q K D hD schedule s b r i hi
        (by decide) (by decide),if_pos (complete b r),
        grid_observe model op "dst" "left" "right" B Q K D hD s b r i hi (by decide) (by decide)]
    · rw [rows_frame model op "dst" "left" "right" B Q K D hD schedule s "dst" offset (Or.inr (by omega)),
        rows_frame model op "dst" "left" "right" B Q K D hD (programs B Q) s "dst" offset (Or.inr (by omega))]
  · rw [rows_frame model op "dst" "left" "right" B Q K D hD schedule s region offset (Or.inl same),
      rows_frame model op "dst" "left" "right" B Q K D hD (programs B Q) s region offset (Or.inl same)]

def candidate {α : Type} [Inhabited α] (model : Algebra α) (op : Operation)
    (B Q K D : Nat) (s : State α) : Option (State α) :=
  KernelExecution.rows model op B Q K D (programs B Q).reverse s

specification reversed_schedule_equivalent {α : Type} [Inhabited α] (model : Algebra α) (op : Operation)
    (B Q K D : Nat) (hD : 0<D) (covered : K≤D) (s : State α) :
    ∃ original reordered,
      KernelExecution.rows model op B Q K D (programs B Q) s=some original ∧
      candidate model op B Q K D s=some reordered ∧ reordered.mem=original.mem := by
  refine ⟨runRows model op "dst" "left" "right" B Q K D hD (programs B Q) s,
    runRows model op "dst" "left" "right" B Q K D hD (programs B Q).reverse s,
    KernelExecution.rows_success _ _ _ _ _ _ hD _ _,KernelExecution.rows_success _ _ _ _ _ _ hD _ _,?_⟩
  exact memory_eq model op B Q K D hD covered _ (by intro b r; simpa using programs_mem B Q b r) s

#axiomsClean coordinate
#axiomsClean memory_eq
#axiomsClean reversed_schedule_equivalent
end
end VeriTile.Bench.Repairs.Ksoftmax.KernelSchedule
