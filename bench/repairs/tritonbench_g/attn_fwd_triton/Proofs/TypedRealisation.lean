import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.TypedPhysical
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.ByteLayout

/-! Opaque full-state realization using the existing shared physical-cell
pullback and the independently proved source frame. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.TypedRealisation
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayerNormFwd
open AliasMemory (Layout Key FreshWrites)
open TypedPhysical
noncomputable section
open scoped Classical
variable {α : Type}
def realise (a : Layout) (initial : Physical α a) (final : State α)
    (writes : Key → Prop) : Physical α a := fun i =>
  if h : ∃ k, writes k ∧ a.address k = i.val then
    final.mem (Classical.choose h).1 (Classical.choose h).2
  else initial i

theorem realise_written (a : Layout) (initial : Physical α a) (final : State α)
    (writes : Key → Prop) (hf : FreshWrites a writes) (k : Key) (hk : writes k) :
    TypedPhysical.read a (realise a initial final writes) k = final.mem k.1 k.2 := by
  simp only [TypedPhysical.read,dif_pos (hf k hk).1,realise]
  have hex : ∃ l, writes l ∧ a.address l = a.address k := ⟨k,hk,rfl⟩
  rw [dif_pos hex]
  have chosen := Classical.choose_spec hex
  have he : Classical.choose hex = k :=
    (hf k hk).2 _ (hf _ chosen.1).1 chosen.2
  rw [he]

theorem realise_unwritten (a : Layout) (initial : Physical α a) (final : State α)
    (writes : Key → Prop) (i : Fin a.capacity)
    (h : ∀ k, writes k → a.address k ≠ i.val) :
    realise a initial final writes i = initial i := by
  apply dif_neg
  rintro ⟨k,hk,ha⟩
  exact h k hk ha

/-- Exact pullback of every live view after a source run with its full frame.
This is the alias-aware physical refinement missing from the disjoint FlatAlloc
bridge. It requires only fresh output writes, not disjoint input allocations. -/
theorem realise_pull (a : Layout) (initial : Physical α a) (s final : State α)
    (writes : Key → Prop) (hf : FreshWrites a writes)
    (hframe : ∀ k, ¬writes k → final.mem k.1 k.2 = (pull a initial s).mem k.1 k.2)
    (k : Key) (hl : a.live k) :
    TypedPhysical.read a (realise a initial final writes) k = final.mem k.1 k.2 := by
  by_cases hw : writes k
  · exact realise_written a initial final writes hf k hw
  · rw [hframe k hw]
    change TypedPhysical.read a (realise a initial final writes) k = TypedPhysical.read a initial k
    simp only [TypedPhysical.read,dif_pos hl]
    apply realise_unwritten
    intro w hw' ha
    have he : k=w := (hf w hw').2 k hl ha.symm
    exact hw (he.symm ▸ hw')

/-- Whole state realization, including out-of-view cells and all registers. -/
theorem realise_pull_state (a : Layout) (initial : Physical α a) (s final : State α)
    (writes : Key → Prop) (hf : FreshWrites a writes)
    (hframe : ∀ k, ¬writes k → final.mem k.1 k.2 = (pull a initial s).mem k.1 k.2) :
    pull a (realise a initial final writes) final = final := by
  apply state_ext
  · funext r o
    change TypedPhysical.read a (realise a initial final writes) (r,o) = final.mem r o
    by_cases hl : a.live (r,o)
    · exact realise_pull a initial s final writes hf hframe (r,o) hl
    · have hw : ¬writes (r,o) := fun h => hl (hf _ h).1
      rw [hframe (r,o) hw]
      simp [pull,TypedPhysical.read,hl]
  · rfl
  · rfl
  · rfl
  · rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.TypedRealisation
