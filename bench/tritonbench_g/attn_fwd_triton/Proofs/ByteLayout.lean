import bench.tritonbench_g.layer_norm_fwd.Proofs.AliasMemory
/-! Shared ordinary byte-start layout and fresh output intervals. No Real or
opaque execution premise occurs in this metadata contract. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.PhysicalCells
open VeriTile.Triton
open VeriTile.Bench.Repairs.LayerNormFwd
noncomputable section
open scoped Classical
structure Layout extends AliasMemory.Layout where
  width : AliasMemory.Key → Nat
  positive : ∀ k, live k → 0 < width k
  boundedCell : ∀ k, live k → address k+width k ≤ capacity

/-- Full byte intervals, so a store cannot overlap the interior of a live
FP32 input cell. Read/read intervals are deliberately unconstrained. -/
def FreshBytes (a : Layout) (writes : AliasMemory.Key → Prop) : Prop :=
  ∀ w, writes w → a.live w ∧ ∀ k, a.live k → k≠w →
    a.address k+a.width k ≤ a.address w ∨ a.address w+a.width w ≤ a.address k

theorem fresh_cells (a : Layout) (writes : AliasMemory.Key → Prop) (h : FreshBytes a writes) :
    AliasMemory.FreshWrites a.toLayout writes := by
  intro w hw
  refine ⟨(h w hw).1,?_⟩
  intro k hk he
  by_contra hne
  have hs := (h w hw).2 k hk hne
  have hk' := a.positive k hk
  have hw' := a.positive w (h w hw).1
  change a.address k=a.address w at he
  rcases hs with hs | hs <;> omega

inductive Role where
  | query | key | value | output | queryScale | keyScale
  deriving DecidableEq

def width : Role → Nat
  | .output => 2
  | _ => 4

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.PhysicalCells
