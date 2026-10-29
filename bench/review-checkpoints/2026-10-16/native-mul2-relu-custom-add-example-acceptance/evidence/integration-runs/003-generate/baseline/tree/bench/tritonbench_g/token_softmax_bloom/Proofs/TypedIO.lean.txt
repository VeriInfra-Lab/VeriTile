import bench.tritonbench_g.token_softmax_bloom.Proofs.TypedContract
namespace VeriTile.Bench.TritonBenchG.TokenSoftmaxBloom.Typed
open VeriTile.Triton FP.Structural
noncomputable section

def Outside {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (r : RegionName) (o : Nat) : Prop :=
  r ≠ p.output ∨ ∀ i : TileIndex [p.width], i.1.val < count p s → o ≠ outputOffset p s i

theorem offset_injective {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (hstride : p.outputTokenStride ≠ 0) : Function.Injective (outputOffset p s) := by
  intro i j h
  simp only [outputOffset, Nat.add_left_cancel_iff] at h
  have hm := Nat.eq_of_mul_eq_mul_right (Nat.pos_of_ne_zero hstride) h
  have hv := Nat.add_left_cancel hm
  have hf : i.1 = j.1 := Fin.ext hv
  cases i; cases j
  simp_all

theorem reference_readback {α : Type} [Inhabited α] (M : Algebra α) (d : FloatDType)
    (p : Parameters) (s : State α) (hstride : p.outputTokenStride ≠ 0)
    (i : TileIndex [p.width]) (hi : i.1.val < count p s) :
    (reference M d p s).mem p.output (outputOffset p s i) =
      .mk d.toTileDType (ofFloat d (outputValues M d p s i)) := by
  exact State.masked_scatter_readback _ _ _ _ _ i hi (offset_injective p s hstride)

theorem reference_frame {α : Type} [Inhabited α] (M : Algebra α) (d : FloatDType)
    (p : Parameters) (s : State α) (r : RegionName) (o : Nat) (h : Outside p s r o) :
    (reference M d p s).mem r o = s.mem r o := by
  apply State.masked_scatter_frame
  rcases h with h | h
  · exact Or.inl h
  · exact Or.inr (fun i _ hi => h i hi)

#axiomsClean offset_injective
#axiomsClean reference_readback
#axiomsClean reference_frame
end
end VeriTile.Bench.TritonBenchG.TokenSoftmaxBloom.Typed
