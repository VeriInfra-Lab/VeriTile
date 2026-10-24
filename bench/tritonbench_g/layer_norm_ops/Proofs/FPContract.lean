import bench.tritonbench_g.layer_norm_ops.Proofs.FPComplete
namespace VeriTile.Bench.TritonBenchG.LayerNormOps.FPStages
open VeriTile.Triton FP.Structural
noncomputable section

theorem scatter_value {α : Type} (p : Parameters) (s t : State α) (r : RegionName)
    (stride : Nat) (values : TileIndex [p.block] → Cell α) (i : TileIndex [p.block])
    (hi : i.1.val<p.columns) : (scatter p s r stride values t).mem r (offset s stride i)=values i := by
  apply State.masked_scatter_readback
  · exact hi
  · intro a b hab
    have hh : a.1.val=b.1.val := Nat.add_left_cancel hab
    rcases a with ⟨a,⟨⟩⟩
    rcases b with ⟨b,⟨⟩⟩
    simpa [Fin.ext_iff] using hh

theorem scatter_frame {α : Type} (p : Parameters) (s t : State α) (r : RegionName)
    (stride : Nat) (values : TileIndex [p.block] → Cell α) (r' : RegionName) (o : Nat)
    (h : OutsideVector p s r stride r' o) : (scatter p s r stride values t).mem r' o=t.mem r' o := by
  apply State.masked_scatter_frame
  rcases h with h | h
  · exact Or.inl h
  · exact Or.inr (fun i _ hi => h i hi)

def RowOutside {α : Type} (p : Parameters) (s : State α) (plain : Bool) (r : RegionName) (o : Nat) :=
  OutsideVector p s p.dx p.dxStride r o ∧ OutsideVector p s p.dw p.columns r o ∧
    (plain=Bool.true → OutsideVector p s p.db p.columns r o)

theorem row_frame {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (plain : Bool) (r : RegionName) (o : Nat) (h : RowOutside p s plain r o) :
    (rowReference M p s plain).mem r o=s.mem r o := by
  cases plain
  · simp only [rowReference, Bool.false_eq_true, ↓reduceIte]
    rw [scatter_frame _ _ _ _ _ _ _ _ h.2.1, scatter_frame _ _ _ _ _ _ _ _ h.1]
  · simp only [rowReference, ↓reduceIte]
    rw [scatter_frame _ _ _ _ _ _ _ _ (h.2.2 rfl),
      scatter_frame _ _ _ _ _ _ _ _ h.2.1, scatter_frame _ _ _ _ _ _ _ _ h.1]

theorem rms_row_values {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (hxd : p.dx≠p.dw) (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (rowReference M p s Bool.false).mem p.dx (offset s p.dxStride i)=stored M (rowDX M p s Bool.false i) ∧
    (rowReference M p s Bool.false).mem p.dw (offset s p.columns i)=stored M (rowDW M p s Bool.false i) := by
  simp only [rowReference, Bool.false_eq_true, ↓reduceIte]
  constructor
  · rw [scatter_frame _ _ _ _ _ _ _ _ (Or.inl hxd)]
    exact scatter_value _ _ _ _ _ _ _ hi
  · exact scatter_value _ _ _ _ _ _ _ hi

theorem plain_row_values {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (hxd : p.dx≠p.dw) (hxb : p.dx≠p.db) (hdb : p.dw≠p.db)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (rowReference M p s Bool.true).mem p.dx (offset s p.dxStride i)=stored M (rowDX M p s Bool.true i) ∧
    (rowReference M p s Bool.true).mem p.dw (offset s p.columns i)=stored M (rowDW M p s Bool.true i) ∧
    (rowReference M p s Bool.true).mem p.db (offset s p.columns i)=stored M (rowDB M p s i) := by
  simp only [rowReference, ↓reduceIte]
  refine ⟨?_,?_,?_⟩
  · rw [scatter_frame _ _ _ _ _ _ _ _ (Or.inl hxb), scatter_frame _ _ _ _ _ _ _ _ (Or.inl hxd)]
    exact scatter_value _ _ _ _ _ _ _ hi
  · rw [scatter_frame _ _ _ _ _ _ _ _ (Or.inl hdb)]
    exact scatter_value _ _ _ _ _ _ _ hi
  · exact scatter_value _ _ _ _ _ _ _ hi

theorem residual_values {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (h : p.dresIn≠p.dx) (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (residualReference M p s).mem p.dx (offset s p.dxStride i)=stored M (residualValue M p s i) ∧
    (residualReference M p s).mem p.dresIn (offset s p.dresInStride i)=stored M (residualValue M p s i) := by
  unfold residualReference
  constructor
  · exact scatter_value _ _ _ _ _ _ _ hi
  · rw [scatter_frame _ _ _ _ _ _ _ _ (Or.inl h)]
    exact scatter_value _ _ _ _ _ _ _ hi

theorem residual_frame {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (r : RegionName) (o : Nat)
    (h : OutsideVector p s p.dx p.dxStride r o ∧ OutsideVector p s p.dresIn p.dresInStride r o) :
    (residualReference M p s).mem r o=s.mem r o := by
  unfold residualReference
  rw [scatter_frame _ _ _ _ _ _ _ _ h.1, scatter_frame _ _ _ _ _ _ _ _ h.2]

#axiomsClean scatter_value
#axiomsClean scatter_frame
#axiomsClean row_frame
#axiomsClean rms_row_values
#axiomsClean plain_row_values
#axiomsClean residual_values
#axiomsClean residual_frame
end
end VeriTile.Bench.TritonBenchG.LayerNormOps.FPStages
