import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.FPForwardComplete
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]
theorem maskedScatter_filter {ι : Type} (r : RegionName) (off : ι → Nat) (v : ι → Cell α)
    (mask : ι → Bool) (l : List ι) (s : State α) :
    l.foldl (fun t i => if mask i then t.write r (off i) (v i) else t) s =
      (l.filter mask).foldl (fun t i => t.write r (off i) (v i)) s := by
  induction l generalizing s with
  | nil => rfl
  | cons i l ih => cases hm : mask i <;> simp [List.foldl_cons,hm,ih]

theorem maskedScatter_frame {ι : Type} (r : RegionName) (off : ι → Nat) (v : ι → Cell α)
    (mask : ι → Bool) (l : List ι) (s : State α) (r' : RegionName) (o : Nat)
    (h : r'≠r ∨ ∀ i ∈ l, mask i=Bool.true → o≠off i) :
    (l.foldl (fun t i => if mask i then t.write r (off i) (v i) else t) s).mem r' o=s.mem r' o := by
  rw [maskedScatter_filter]
  apply State.scatter_frame
  rcases h with h|h
  · exact Or.inl h
  · exact Or.inr (by intro i hi; exact h i (List.mem_filter.mp hi).1 (List.mem_filter.mp hi).2)

theorem maskedScatter_readback (r : RegionName) (off : TileIndex shape → Nat) (v : TileIndex shape → Cell α)
    (mask : TileIndex shape → Bool) (s : State α) (i : TileIndex shape)
    (ha : mask i=Bool.true) (hinj : Function.Injective off) :
    ((TileShape.allIndices shape).foldl (fun t i => if mask i then t.write r (off i) (v i) else t) s).mem r (off i)=v i := by
  rw [maskedScatter_filter]
  exact State.scatter_readback_list _ _ _ _ _ i
    ((TileShape.allIndices_nodup shape).filter mask)
    (List.mem_filter.mpr ⟨TileShape.mem_allIndices _ i,ha⟩) hinj

theorem forwardRow_injective (p : ForwardParameters) (s : State α) : Function.Injective (forwardRowOffset p s) := by
  intro a b he
  apply Prod.ext
  · apply Fin.ext
    simpa only [forwardRowOffset,forwardQueryRow,Nat.add_left_cancel_iff] using he
  · exact Subsingleton.elim _ _

theorem forwardCorrectness (rest : Algebra α) (P : Primitives α) (p : ForwardParameters)
    (initial : State α) (hn : 0<p.blockN) (hd : p.blockN ∣ forwardStop p initial)
    (hLO : p.denominator≠p.output) (hMO : p.maximum≠p.output) (hML : p.maximum≠p.denominator)
    (hinj : Function.Injective (forwardOffset p initial)) :
    ∃ final, FP.Structural.exec (backend rest P) (forwardSource p) initial=some final ∧
      final.mem=(forwardWrite P p initial initial).mem ∧
      (∀ i, (forwardOutPointer p initial).inBounds (TileShape.indexToList _ i) [0,1]=Bool.true →
        final.mem p.output (forwardOffset p initial i)=forwardOutput P p initial i) ∧
      (∀ i, final.mem p.denominator (forwardRowOffset p initial i)=forwardLCell P p initial i) ∧
      (∀ i, final.mem p.maximum (forwardRowOffset p initial i)=forwardMCell P p initial i) ∧
      (∀ r o, (r≠p.output ∨ ∀ i, (forwardOutPointer p initial).inBounds (TileShape.indexToList _ i) [0,1]=Bool.true → o≠forwardOffset p initial i) →
        (r≠p.denominator ∨ ∀ i, o≠forwardRowOffset p initial i) →
        (r≠p.maximum ∨ ∀ i, o≠forwardRowOffset p initial i) → final.mem r o=initial.mem r o) := by
  obtain ⟨final,he,hm⟩ := forwardFullExecution rest P p initial hn hd
  refine ⟨final,he,hm,?_,?_,?_,?_⟩
  · intro i ha
    rw [hm]
    exact maskedScatter_readback _ _ _ _ _ i ha hinj
  · intro i
    rw [hm]
    unfold forwardWrite
    rw [maskedScatter_frame _ _ _ _ _ _ _ _ (Or.inl hLO)]
    rw [State.scatter_frame _ _ _ _ _ _ (Or.inl (Ne.symm hML))]
    exact State.scatter_readback _ _ _ _ i (forwardRow_injective p initial)
  · intro i
    rw [hm]
    unfold forwardWrite
    rw [maskedScatter_frame _ _ _ _ _ _ _ _ (Or.inl hMO)]
    exact State.scatter_readback _ _ _ _ i (forwardRow_injective p initial)
  · intro r o ho hl hm'
    rw [hm]
    unfold forwardWrite
    rw [maskedScatter_frame _ _ _ _ _ _ _ _ (ho.imp id (fun h i _ => h i))]
    rw [State.scatter_frame _ _ _ _ _ _ (hm'.imp id (fun h i _ => h i))]
    exact State.scatter_frame _ _ _ _ _ _ (hl.imp id (fun h i _ => h i)) _

#axiomsClean forwardCorrectness
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
