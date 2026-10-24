import bench.tritonbench_g.rotary_transform.Proofs.FPRow

namespace VeriTile.Bench.TritonBenchG.RotaryTransform.FPRow
open VeriTile.Triton FP.Structural
open KcacheCopyTriton.TypedCopy (masked_readback_active)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000

/-- Both half-row address maps preserve lane identity under the original
nonzero component-stride condition. No numerical premise is involved. -/
theorem output_injective {α : Type} [Inhabited α] (k : Kind) (p : Parameters)
    (s : State α) (second : Bool) (hstride : p.strideOutDimension ≠ 0) :
    Function.Injective (outputOffset k p s second) := by
  intro i j h
  have hm := Nat.add_left_cancel h
  have ha := (Nat.mul_left_inj hstride).mp hm
  exact Prod.ext (Fin.ext (Nat.add_right_cancel ha)) (Subsingleton.elim _ _)

/-- Disjointness uses active-lane masks, so even a padded tile wider than the
rotary half dimension introduces no additional restriction. -/
theorem halves_disjoint {α : Type} [Inhabited α] (k : Kind) (p : Parameters)
    (s : State α) (hstride : p.strideOutDimension ≠ 0)
    (i j : TileIndex [p.width]) (hi : active k p s i) (hj : active k p s j) :
    outputOffset k p s Bool.false i ≠ outputOffset k p s Bool.true j := by
  intro h
  have hm := Nat.add_left_cancel h
  have ha := (Nat.mul_left_inj hstride).mp hm
  simp only [Bool.false_eq_true, ↓reduceIte, Nat.add_zero] at ha
  have hil := hi.2
  omega

theorem readback {α : Type} [Inhabited α] (M : Algebra α) (k : Kind)
    (p : Parameters) (s : State α) (hstride : p.strideOutDimension ≠ 0)
    (second : Bool) (i : TileIndex [p.width]) (hi : active k p s i) :
    (execution M k p s).val.mem p.output (outputOffset k p s second i) =
      .mk .real (value M k p s second i) := by
  rw [execution_memory]
  cases second
  · unfold copied halfScatter
    rw [State.masked_scatter_frame _ _ _ _ _ _ _ (Or.inr (by
      intro j _ hj
      exact halves_disjoint k p s hstride i j hi hj))]
    exact State.masked_scatter_readback _ _ _ _ _ i hi
      (output_injective k p s Bool.false hstride)
  · exact State.masked_scatter_readback _ _ _ _ _ i hi
      (output_injective k p s Bool.true hstride)

def Outside {α : Type} [Inhabited α] (k : Kind) (p : Parameters) (s : State α)
    (r : RegionName) (o : Nat) : Prop :=
  r ≠ p.output ∨ ∀ second (i : TileIndex [p.width]), active k p s i →
    o ≠ outputOffset k p s second i

theorem frame {α : Type} [Inhabited α] (M : Algebra α) (k : Kind)
    (p : Parameters) (s : State α) (r : RegionName) (o : Nat) (h : Outside k p s r o) :
    (execution M k p s).val.mem r o = s.mem r o := by
  rw [execution_memory]
  unfold copied halfScatter
  rw [State.masked_scatter_frame _ _ _ _ _ _ _
    (h.imp id (fun h i _ hi => h Bool.true i hi))]
  rw [State.masked_scatter_frame _ _ _ _ _ _ _
    (h.imp id (fun h i _ hi => h Bool.false i hi))]

def Post {α : Type} [Inhabited α] (k : Kind) (p : Parameters) (initial final : State α)
    (results : Bool → TileIndex [p.width] → α) : Prop :=
  (∀ second i, active k p initial i →
    final.mem p.output (outputOffset k p initial second i) = .mk .real (results second i)) ∧
  (∀ r o, Outside k p initial r o → final.mem r o = initial.mem r o)

theorem correctness {α : Type} [Inhabited α] (M : Algebra α) (k : Kind)
    (p : Parameters) (s : State α) (hstride : p.strideOutDimension ≠ 0) :
    ∃ t, FP.Structural.exec M (source k p) s = some t ∧ Post k p s t (value M k p s) :=
  ⟨(execution M k p s).val, (execution M k p s).property,
    fun second i hi => readback M k p s hstride second i hi, frame M k p s⟩

#axiomsClean output_injective
#axiomsClean halves_disjoint
#axiomsClean readback
#axiomsClean frame
#axiomsClean correctness
end
end VeriTile.Bench.TritonBenchG.RotaryTransform.FPRow
