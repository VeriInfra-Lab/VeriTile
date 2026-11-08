import bench.tritonbench_g.token_softmax_llama.Proofs.Model
namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
open VeriTile.Triton FP.Structural
noncomputable section

def Outside {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (r : RegionName) (o : Nat) : Prop :=
  r ≠ p.output ∨ ∀ i : TileIndex [p.width], i.1.val  <  count p s → o ≠ outputOffset p s i

/-- Nonnegative segment starts were implicit in the former Nat metadata
carrier. The actual signed port states that condition here; complete-grid
proofs instead use their existing exact event-uniqueness condition. -/
theorem offset_injective {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (hstride : p.outputTokenStride ≠ 0) (hstart : 0 ≤ startIndex p s) : Function.Injective (outputOffset p s) := by
  intro i j h
  have hi : 0 ≤ ((s.pids 1 : Int)*p.outputHeadStride +
      (startIndex p s + i.1.val)*p.outputTokenStride) := by positivity
  have hj : 0 ≤ ((s.pids 1 : Int)*p.outputHeadStride +
      (startIndex p s + j.1.val)*p.outputTokenStride) := by positivity
  unfold outputOffset at h
  have he : ((s.pids 1 : Int)*p.outputHeadStride + (startIndex p s + i.1.val)*p.outputTokenStride) =
      ((s.pids 1 : Int)*p.outputHeadStride + (startIndex p s + j.1.val)*p.outputTokenStride) := by omega
  have hp : (0 : Int) < p.outputTokenStride := by exact_mod_cast Nat.pos_of_ne_zero hstride
  have hv : i.1.val = j.1.val := by nlinarith
  exact Prod.ext (Fin.ext hv) (Subsingleton.elim _ _)

theorem reference_readback {α : Type} [Inhabited α] (M : Algebra α) (d : FloatDType)
    (p : Parameters) (s : State α) (hstride : p.outputTokenStride ≠ 0) (hstart : 0 ≤ startIndex p s)
    (i : TileIndex [p.width]) (hi : i.1.val  <  count p s) :
    (reference M d p s).mem p.output (outputOffset p s i) =
      .mk d.toTileDType (ofFloat d (outputValues M d p s i)) := by
  exact State.masked_scatter_readback _ _ _ _ _ i hi (offset_injective p s hstride hstart)

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
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
