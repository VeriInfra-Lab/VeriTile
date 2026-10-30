import bench.tritonbench_g.triton_attention.Proofs.FPBackwardComplete
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]
def backwardWritten (p : BackwardParameters) (s : State α) (r : RegionName) (o : Nat) : Prop :=
  (r=p.dq ∧ ∃ c<p.blocks, ∃ i : BLane p, o=backwardOffset p s c i) ∨
  ((r=p.dk ∨ r=p.dv) ∧ ∃ c<p.blocks, ∃ i : BLane p,
    (backwardPointer p s r c).inBounds (TileShape.indexToList _ i) [0,1]=Bool.true ∧ o=backwardOffset p s c i)

theorem memoryFold_frame {ι : Type} (l : List ι) (r : RegionName) (off : ι → Nat) (v : ι → Cell α)
    (mem : Memory α) (r' : RegionName) (o : Nat) (h : ∀ i ∈ l, r'≠r ∨ o≠off i) :
    (l.foldl (fun m i => writeMemory m r (off i) (v i)) mem) r' o=mem r' o := by
  induction l generalizing mem with
  | nil => rfl
  | cons i l ih =>
    rw [List.foldl_cons,ih _ (by intro k hk; exact h k (List.mem_cons_of_mem _ hk))]
    rcases h i (by simp) with h|h <;> simp [writeMemory,h]

theorem memoryMaskedFold_frame {ι : Type} (l : List ι) (r : RegionName) (off : ι → Nat) (v : ι → Cell α)
    (mask : ι → Bool) (mem : Memory α) (r' : RegionName) (o : Nat)
    (h : ∀ i ∈ l, mask i=Bool.true → r'≠r ∨ o≠off i) :
    (l.foldl (fun m i => if mask i then writeMemory m r (off i) (v i) else m) mem) r' o=mem r' o := by
  induction l generalizing mem with
  | nil => rfl
  | cons i l ih =>
    rw [List.foldl_cons,ih _ (by intro k hk; exact h k (List.mem_cons_of_mem _ hk))]
    cases hm : mask i
    · simp [hm]
    · rcases h i (by simp) hm with h|h <;> simp [hm,writeMemory,h]

theorem backwardInner_frame (P : Primitives α) (p : BackwardParameters) (s : State α) (mem : Memory α)
    (c n : Nat) (hn : c+n≤p.blocks) (r : RegionName) (o : Nat) (h : ¬backwardWritten p s r o) :
    (backwardInnerRounds P p s mem c n).mem r o=mem r o := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [backwardInnerRounds]
    change ((TileShape.allIndices [p.blockM,p.head]).foldl (fun m i => writeMemory m p.dq (backwardOffset p s (c+n) i) _) _) r o=_
    rw [memoryFold_frame]
    · exact ih (by omega)
    · intro i _
      by_cases hr : r=p.dq
      · right
        intro ho
        exact h (Or.inl ⟨hr,c+n,by omega,i,ho⟩)
      · exact Or.inl hr

theorem backwardOuter_frame (P : Primitives α) (p : BackwardParameters) (s : State α) (mem : Memory α)
    (c : Nat) (hc : c<p.blocks) (r : RegionName) (o : Nat) (h : ¬backwardWritten p s r o) :
    backwardOuterAdvance P p s c mem r o=mem r o := by
  unfold backwardOuterAdvance
  rw [memoryMaskedFold_frame]
  · rw [memoryMaskedFold_frame]
    · exact backwardInner_frame P p s mem c _ (by omega) r o h
    · intro i _ ha
      by_cases hr : r=p.dv
      · right
        intro ho
        apply h
        exact Or.inr ⟨Or.inr hr,c,hc,i,by simpa only [hr] using ha,ho⟩
      · exact Or.inl hr
  · intro i _ ha
    by_cases hr : r=p.dk
    · right
      intro ho
      apply h
      exact Or.inr ⟨Or.inl hr,c,hc,i,by simpa only [hr] using ha,ho⟩
    · exact Or.inl hr

theorem backwardFrame (P : Primitives α) (p : BackwardParameters) (s : State α)
    (r : RegionName) (o : Nat) (h : ¬backwardWritten p s r o) :
    backwardFinalMemory P p s r o=s.mem r o := by
  have aux : ∀ n, n≤p.blocks → backwardRounds P p s n r o=s.mem r o := by
    intro n
    induction n with
    | zero => intro _; rfl
    | succ n ih =>
      intro hn
      rw [backwardRounds,backwardOuter_frame P p s _ n (by omega) r o h]
      exact ih (by omega)
  exact aux _ (Nat.le_refl _)

theorem backwardCorrectness (rest : Algebra α) (P : Primitives α) (p : BackwardParameters)
    (initial : State α) (hBM : 0<p.blockM) :
    ∃ final, FP.Structural.exec (backend rest P) (backwardSource p) initial=some final ∧
      final.mem=backwardFinalMemory P p initial ∧
      (∀ c<p.blocks, ∀ i : BLane p,
        final.mem p.dq (backwardOffset p initial c i)=backwardFinalMemory P p initial p.dq (backwardOffset p initial c i) ∧
        final.mem p.dk (backwardOffset p initial c i)=backwardFinalMemory P p initial p.dk (backwardOffset p initial c i) ∧
        final.mem p.dv (backwardOffset p initial c i)=backwardFinalMemory P p initial p.dv (backwardOffset p initial c i)) ∧
      (∀ r o, ¬backwardWritten p initial r o → final.mem r o=initial.mem r o) := by
  obtain ⟨final,he,hm⟩ := backwardFullExecution rest P p initial hBM
  refine ⟨final,he,hm,?_,?_⟩
  · simp only [hm,implies_true,and_self]
  · intro r o h
    rw [hm]
    exact backwardFrame P p initial r o h

#axiomsClean backwardFrame
#axiomsClean backwardCorrectness
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
