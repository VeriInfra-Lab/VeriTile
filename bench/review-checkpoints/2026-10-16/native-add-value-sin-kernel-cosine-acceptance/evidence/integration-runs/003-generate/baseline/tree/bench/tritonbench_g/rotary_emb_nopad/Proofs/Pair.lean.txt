import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy
namespace VeriTile.Bench.TritonBenchG.RotaryEmbNopad.Pair
open VeriTile.Triton FP.Structural
open KcacheCopyTriton.TypedCopy (masked_readback_active)
noncomputable section

def halfScatter {α : Type} (r : RegionName) (active : TileIndex shape → Prop)
    [DecidablePred active] (off : Bool → TileIndex shape → Nat)
    (value : Bool → TileIndex shape → Cell α) (second : Bool) (s : State α) : State α :=
  (TileShape.allIndices shape).foldl
    (fun t i => if active i then t.write r (off second i) (value second i) else t) s

def written {α : Type} (r : RegionName) (active : TileIndex shape → Prop)
    [DecidablePred active] (off : Bool → TileIndex shape → Nat)
    (value : Bool → TileIndex shape → Cell α) (s : State α) : State α :=
  halfScatter r active off value Bool.true (halfScatter r active off value Bool.false s)

def WritesApart (active : TileIndex shape → Prop) (off : Bool → TileIndex shape → Nat) : Prop :=
  (∀ second a b, active a → active b → off second a = off second b → a = b) ∧
  (∀ a b, active a → active b → off Bool.false a ≠ off Bool.true b)

theorem readback {α : Type} (r : RegionName) (active : TileIndex shape → Prop)
    [DecidablePred active] (off : Bool → TileIndex shape → Nat)
    (value : Bool → TileIndex shape → Cell α) (s : State α)
    (hw : WritesApart active off) (second : Bool) (i : TileIndex shape) (hi : active i) :
    (written r active off value s).mem r (off second i) = value second i := by
  cases second
  · unfold written halfScatter
    rw [State.masked_scatter_frame _ _ _ _ _ _ _ (Or.inr (by
      intro j _ hj
      exact hw.2 i j hi hj))]
    exact masked_readback_active _ _ _ _ _ i hi (hw.1 Bool.false)
  · exact masked_readback_active _ _ _ _ _ i hi (hw.1 Bool.true)

def Outside (region : RegionName) (active : TileIndex shape → Prop)
    (off : Bool → TileIndex shape → Nat) (r : RegionName) (o : Nat) : Prop :=
  r ≠ region ∨ ∀ second i, active i → o ≠ off second i

theorem frame {α : Type} (region : RegionName) (active : TileIndex shape → Prop)
    [DecidablePred active] (off : Bool → TileIndex shape → Nat)
    (value : Bool → TileIndex shape → Cell α) (s : State α) (r : RegionName) (o : Nat)
    (h : Outside region active off r o) :
    (written region active off value s).mem r o = s.mem r o := by
  unfold written halfScatter
  rw [State.masked_scatter_frame _ _ _ _ _ _ _ (h.imp id (fun h i _ hi => h Bool.true i hi))]
  rw [State.masked_scatter_frame _ _ _ _ _ _ _ (h.imp id (fun h i _ hi => h Bool.false i hi))]

#axiomsClean readback
#axiomsClean frame
end
end VeriTile.Bench.TritonBenchG.RotaryEmbNopad.Pair
