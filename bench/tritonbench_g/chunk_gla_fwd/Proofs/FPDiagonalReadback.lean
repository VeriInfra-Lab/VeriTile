import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPDiagonalLoop
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.FPDiagonal
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_readback_active)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000

theorem output_injective (split : Bool) (c : Layout) (s : State α)
    (hwidth : c.BC ≤ c.BT) (j k : Nat) (hj : j<c.BC) (hk : k<c.BC)
    (i l : Index c) (he : outputAddress split c s j i=outputAddress split c s k l) :
    i=l ∧ j=k := by
  have he' : i.1.val*(if split then c.BC else c.BT)+j=
      l.1.val*(if split then c.BC else c.BT)+k := by
    cases split <;> simp [outputAddress,outputBase,row,Nat.add_mul] at he ⊢ <;> omega
  have hj' : j<(if split then c.BC else c.BT) := by cases split <;> simp_all <;> omega
  have hk' : k<(if split then c.BC else c.BT) := by cases split <;> simp_all <;> omega
  obtain ⟨hi,hj⟩ := Geometry.packed_injective hj' hk' he'
  exact ⟨Prod.ext (Fin.ext hi) (Subsingleton.elim _ _),hj⟩

theorem columnReadback (M : Algebra α) (split : Bool) (c : Layout) (scale : ℝ)
    (initial current : State α) (j : Nat) (i : Index c) (hi : active split c initial i)
    (hw : ∀ a b, active split c initial a → active split c initial b →
      outputAddress split c initial j a=outputAddress split c initial j b → a=b) :
    (columnWritten M split c scale initial j current).mem (out split) (outputAddress split c initial j i)=
      .mk .real (value M split c scale initial j i) :=
  masked_readback_active _ _ _ _ _ i hi hw

theorem foldFrame (M : Algebra α) (split : Bool) (c : Layout) (scale : ℝ)
    (initial current : State α) (start stop : Nat) (r : RegionName) (off : Nat)
    (hn : r≠out split ∨ ∀ j, start ≤ j → j<stop →
      ∀ i, active split c initial i → off≠outputAddress split c initial j i) :
    (foldRange (columnWritten M split c scale initial) start stop 1 current).mem r off=current.mem r off := by
  rw [foldRange]
  simp only [Nat.one_ne_zero,↓reduceDIte]
  split
  · rename_i hlt
    rw [foldFrame M split c scale initial (columnWritten M split c scale initial start current)
      (start+1) stop r off (by
        rcases hn with hr|ho
        · exact Or.inl hr
        · exact Or.inr (fun j hj hj' i hi => ho j (by omega) hj' i hi))]
    apply columnFrame
    rcases hn with hr|ho
    · exact Or.inl hr
    · exact Or.inr (ho start (by omega) hlt)
  · rfl
termination_by stop-start
decreasing_by omega

theorem foldReadback (M : Algebra α) (split : Bool) (c : Layout) (scale : ℝ)
    (initial current : State α) (start stop j : Nat) (i : Index c)
    (hi : active split c initial i) (hstart : start ≤ j) (hj : j<stop)
    (hw : ∀ j k, j<stop → k<stop → ∀ a b,
      active split c initial a → active split c initial b →
      outputAddress split c initial j a=outputAddress split c initial k b → a=b ∧ j=k) :
    (foldRange (columnWritten M split c scale initial) start stop 1 current).mem (out split)
      (outputAddress split c initial j i)=.mk .real (value M split c scale initial j i) := by
  rw [foldRange]
  have hlt : start<stop := by omega
  simp only [Nat.one_ne_zero,↓reduceDIte,hlt]
  by_cases he : start=j
  · subst start
    rw [foldFrame M split c scale initial (columnWritten M split c scale initial j current)
      (j+1) stop (out split) (outputAddress split c initial j i) (Or.inr (by
        intro k hk hk' a ha heq
        have := (hw j k hj hk' i a hi ha heq).2
        omega))]
    exact columnReadback M split c scale initial current j i hi
      (fun a b ha hb hab => (hw j j hj hj a b ha hb hab).1)
  · exact foldReadback M split c scale initial (columnWritten M split c scale initial start current)
      (start+1) stop j i hi (by omega) hj hw
termination_by stop-start
decreasing_by omega

theorem readback (M : Algebra α) (split : Bool) (c : Layout) (scale : ℝ)
    (s : State α) (hwidth : c.BC ≤ c.BT) (j : Nat) (i : Index c)
    (hi : active split c s i) (hj : j<columns split c s) :
    (written M split c scale s).mem (out split) (outputAddress split c s j i)=
      .mk .real (value M split c scale s j i) := by
  apply foldReadback M split c scale s s 0 (columns split c s) j i hi (Nat.zero_le _) hj
  intro j k hj hk a b ha hb he
  exact output_injective split c s hwidth j k
    (lt_of_lt_of_le hj (Nat.min_le_left _ _)) (lt_of_lt_of_le hk (Nat.min_le_left _ _)) a b he

theorem frame (M : Algebra α) (split : Bool) (c : Layout) (scale : ℝ)
    (s : State α) (r : RegionName) (off : Nat)
    (hn : r≠out split ∨ ∀ j, j<columns split c s →
      ∀ i, active split c s i → off≠outputAddress split c s j i) :
    (written M split c scale s).mem r off=s.mem r off := by
  apply foldFrame
  rcases hn with hr|ho
  · exact Or.inl hr
  · exact Or.inr (fun j _ hj => ho j hj)

#axiomsClean readback
#axiomsClean frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.FPDiagonal
