import bench.tritonbench_g.fused_rotary_embedding.Proofs.FPQ
namespace VeriTile.Bench.TritonBenchG.FusedRotaryEmbedding.FPQ
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000

theorem offset_injective {α : Type} (p : Parameters) (s : State α) (second : Bool)
    (hstride : p.dimensionStride ≠ 0) : Function.Injective (offset p s second) := by
  intro i j h
  have ha := (Nat.mul_left_inj hstride).mp (Nat.add_left_cancel h)
  exact Prod.ext (Fin.ext (Nat.add_right_cancel ha)) (Subsingleton.elim _ _)

theorem halves_disjoint {α : Type} (p : Parameters) (s : State α)
    (hstride : p.dimensionStride ≠ 0) (i j : TileIndex [p.half]) :
    offset p s Bool.false i ≠ offset p s Bool.true j := by
  intro h
  have ha := (Nat.mul_left_inj hstride).mp (Nat.add_left_cancel h)
  simp only [Bool.false_eq_true, ↓reduceIte, Nat.add_zero] at ha
  have hil := i.1.isLt
  omega

theorem readback {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (hstride : p.dimensionStride ≠ 0) (second : Bool) (i : TileIndex [p.half]) :
    (execution M p s).val.mem p.q (offset p s second i) = .mk .fp32 (value M p s second i) := by
  rw [execution_memory]
  cases second
  · unfold written halfScatter
    rw [State.scatter_frame _ _ _ _ _ _ (Or.inr (by
      intro j _
      exact halves_disjoint p s hstride i j))]
    exact State.scatter_readback _ _ _ _ i (offset_injective p s Bool.false hstride)
  · exact State.scatter_readback _ _ _ _ i (offset_injective p s Bool.true hstride)

def Outside {α : Type} (p : Parameters) (s : State α) (r : RegionName) (o : Nat) : Prop :=
  r ≠ p.q ∨ ∀ second (i : TileIndex [p.half]), o ≠ offset p s second i

theorem frame {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (r : RegionName) (o : Nat) (h : Outside p s r o) :
    (execution M p s).val.mem r o = s.mem r o := by
  rw [execution_memory]
  unfold written halfScatter
  rw [State.scatter_frame _ _ _ _ _ _ (h.imp id (fun h i _ => h Bool.true i))]
  rw [State.scatter_frame _ _ _ _ _ _ (h.imp id (fun h i _ => h Bool.false i))]

def Post {α : Type} (p : Parameters) (initial final : State α)
    (results : Bool → TileIndex [p.half] → α) : Prop :=
  (∀ second i, final.mem p.q (offset p initial second i) = .mk .fp32 (results second i)) ∧
  (∀ r o, Outside p initial r o → final.mem r o = initial.mem r o)

theorem correctness {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (hstride : p.dimensionStride ≠ 0) :
    ∃ t, FP.Structural.exec M (source p) s = some t ∧ Post p s t (value M p s) :=
  ⟨(execution M p s).val, (execution M p s).property,
    readback M p s hstride, frame M p s⟩

#axiomsClean readback
#axiomsClean frame
#axiomsClean correctness
end
end VeriTile.Bench.TritonBenchG.FusedRotaryEmbedding.FPQ
