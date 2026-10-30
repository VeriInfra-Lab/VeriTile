import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.TypedContract
namespace VeriTile.Bench.Repairs.RopeBackwardTransform.Source
open VeriTile.Triton FP.Structural
noncomputable section

theorem scatter_frame {α : Type} (mem : Memory α) (target : RegionName)
    (pid stride heads hd : Nat) (second : Bool) (H W : Nat)
    (value : TileIndex [H,W] → Cell α) (r : RegionName) (o : Nat)
    (h : r ≠ target ∨ ∀ i : TileIndex [H,W], active heads (hd/2) i →
      o ≠ address pid stride hd (hd/2) second i) :
    scatter mem target pid stride heads hd second H W value r o = mem r o := by
  unfold scatter
  generalize TileShape.allIndices [H,W] = xs
  induction xs generalizing mem with
  | nil => rfl
  | cons i xs ih =>
    simp only [List.foldl_cons, ih]
    by_cases hi : active heads (hd/2) i
    · have hn : ¬ (r = target ∧ o = address pid stride hd (hd/2) second i) := by
        rcases h with h | h
        · exact fun he => h he.1
        · exact fun he => h i hi he.2
      simp only [hi, ↓reduceIte, hn]
    · simp only [hi, Bool.false_eq_true, ↓reduceIte]

/-- The four masked output windows. All other memory remains unchanged. -/
def Outside (p : Parameters) {α : Type} (s : State α) (r : RegionName) (o : Nat) : Prop :=
  (∀ second, r ≠ p.q ∨ ∀ i : TileIndex [p.paddedQHeads,p.paddedHeadDim/2],
    active p.qHeads (p.headDim/2) i →
    o ≠ address (s.pids 0) p.qStride p.headDim (p.headDim/2) second i) ∧
  (∀ second, r ≠ p.k ∨ ∀ i : TileIndex [p.paddedKHeads,p.paddedHeadDim/2],
    active p.kHeads (p.headDim/2) i →
    o ≠ address (s.pids 0) p.kStride p.headDim (p.headDim/2) second i)

theorem reference_frame {α : Type} [Inhabited α] (M : Algebra α)
    (input rotation : FloatDType) (p : Parameters) (s : State α)
    (r : RegionName) (o : Nat) (h : Outside p s r o) :
    referenceMemory M input rotation p s r o = s.mem r o := by
  dsimp only [referenceMemory]
  rw [scatter_frame _ _ _ _ _ _ _ _ _ _ _ _ (h.2 Bool.true)]
  rw [scatter_frame _ _ _ _ _ _ _ _ _ _ _ _ (h.2 Bool.false)]
  rw [scatter_frame _ _ _ _ _ _ _ _ _ _ _ _ (h.1 Bool.true)]
  exact scatter_frame _ _ _ _ _ _ _ _ _ _ _ _ (h.1 Bool.false)

#axiomsClean scatter_frame
#axiomsClean reference_frame
end
end VeriTile.Bench.Repairs.RopeBackwardTransform.Source
