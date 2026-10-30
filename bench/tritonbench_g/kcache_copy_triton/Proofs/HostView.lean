import bench.tritonbench_g.kcache_copy_triton.Proofs.HostViewShape
import bench.tritonbench_g.kcache_copy_triton.Proofs.HostViewCases

/-! Parameterized address preservation for the logical rank4→rank3 view
algorithm. All Nat dimensions and strides are admitted. Singleton coordinates
are zero; a zero dimension makes the indexed domain empty. No native Torch
refinement, storage allocation, signed strides or physical aliasing is claimed. -/
namespace VeriTile.Bench.TritonBenchG.KcacheCopyTriton.Host.View
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 1000000

/-- The row-major merge of the two leading coordinates is a valid coordinate
of the actual flattened dimension, including arbitrary positive batch sizes. -/
theorem flattened_index_lt {a b : Nat} (ia : Fin a) (ib : Fin b) :
    ia.val*b + ib.val < a*b := by
  calc
    ia.val*b + ib.val < ia.val*b + b := Nat.add_lt_add_left ib.isLt _
    _ = (ia.val+1)*b := by ring
    _ ≤ a*b := Nat.mul_le_mul_right b (Nat.succ_le_of_lt ia.isLt)

/-- Successful stride inference preserves every indexed address. In particular
zero/broadcast strides require no separate injectivity assumption. -/
theorem rank4_view_address (a b h d sa sb sh sd st oh od : Nat)
    (view : inferViewStrides [a,b,h,d] [sa,sb,sh,sd] [a*b,h,d] = some [st,oh,od])
    (ia : Fin a) (ib : Fin b) (ih : Fin h) (id : Fin d) :
    ia.val*sa + ib.val*sb + ih.val*sh + id.val*sd =
      (ia.val*b + ib.val)*st + ih.val*oh + id.val*od := by
  have ha : 0 < a := lt_of_le_of_lt (Nat.zero_le ia.val) ia.isLt
  have hb : 0 < b := lt_of_le_of_lt (Nat.zero_le ib.val) ib.isLt
  have hh : 0 < h := lt_of_le_of_lt (Nat.zero_le ih.val) ih.isLt
  have hd : 0 < d := lt_of_le_of_lt (Nat.zero_le id.val) id.isLt
  obtain ⟨ca,cb,ch,cd⟩ := coefficients_positive a b h d sa sb sh sd st oh od ha hb hh hd view
  have ea : ia.val*sa = ia.val*(b*st) := by
    rcases ca with ca | ca
    · have iz : ia.val = 0 := by have := ia.isLt; omega
      simp [iz]
    · rw [ca]
  have eb : ib.val*sb = ib.val*st := by
    rcases cb with cb | cb
    · have iz : ib.val = 0 := by have := ib.isLt; omega
      simp [iz]
    · rw [cb]
  have eh : ih.val*sh = ih.val*oh := by
    rcases ch with ch | ch
    · have iz : ih.val = 0 := by have := ih.isLt; omega
      simp [iz]
    · rw [ch]
  have ed : id.val*sd = id.val*od := by
    rcases cd with cd | cd
    · have iz : id.val = 0 := by have := id.isLt; omega
      simp [iz]
    · rw [cd]
  rw [ea,eb,eh,ed]
  ring

/-- The actual returned list has rank three and preserves all addresses,
including when the source tensor has an empty index domain. -/
theorem rank4_view_addresses (a b h d sa sb sh sd : Nat) (strides : List Nat)
    (view : inferViewStrides [a,b,h,d] [sa,sb,sh,sd] [a*b,h,d] = some strides) :
    ∃ st oh od, strides = [st,oh,od] ∧
      ∀ (ia : Fin a) (ib : Fin b) (ih : Fin h) (id : Fin d),
        ia.val*sa + ib.val*sb + ih.val*sh + id.val*sd =
          (ia.val*b + ib.val)*st + ih.val*oh + id.val*od := by
  obtain ⟨st,oh,od,rfl⟩ := List.length_eq_three.mp
    (rank4_view_length a b h d sa sb sh sd strides view)
  exact ⟨st,oh,od,rfl,rank4_view_address a b h d sa sb sh sd st oh od view⟩

/-- Connect the address result to `reshape` itself. The returned metadata and
region are derived from the successful view branch, rather than assumed as a
separate address mapping. The copied branch is intentionally a separate case. -/
theorem reshape_view_addresses (region fresh : RegionName)
    (a b h d sa sb sh sd : Nat) (result : Reshape)
    (success : reshape ⟨region,[a,b,h,d],[sa,sb,sh,sd]⟩ fresh = .ok result)
    (isView : result.copied = Bool.false) :
    ∃ st oh od, result.tensor = ⟨region,[a*b,h,d],[st,oh,od]⟩ ∧
      ∀ (ia : Fin a) (ib : Fin b) (ih : Fin h) (id : Fin d),
        ia.val*sa + ib.val*sb + ih.val*sh + id.val*sd =
          (ia.val*b + ib.val)*st + ih.val*oh + id.val*od := by
  simp only [reshape] at success
  split at success
  · contradiction
  · cases view : inferViewStrides [a,b,h,d] [sa,sb,sh,sd] [a*b,h,d] with
    | none =>
      simp only [view] at success
      cases success
      contradiction
    | some strides =>
      simp only [view] at success
      cases success
      obtain ⟨st,oh,od,hs,ha⟩ := rank4_view_addresses a b h d sa sb sh sd strides view
      exact ⟨st,oh,od,by simp only [hs],ha⟩

/-- Any read function sees the same value at corresponding indexed addresses
of the successful view, with an arbitrary common storage offset. This is a
logical read equality and does not posit a native allocator or Torch bridge. -/
theorem reshape_view_read {Value : Type} (load : RegionName → Nat → Value)
    (region fresh : RegionName) (offset : Nat)
    (a b h d sa sb sh sd : Nat) (result : Reshape)
    (success : reshape ⟨region,[a,b,h,d],[sa,sb,sh,sd]⟩ fresh = .ok result)
    (isView : result.copied = Bool.false)
    (ia : Fin a) (ib : Fin b) (ih : Fin h) (id : Fin d) :
    load region (offset + (ia.val*sa + ib.val*sb + ih.val*sh + id.val*sd)) =
      load result.tensor.region (offset +
        ((ia.val*b + ib.val)*result.tensor.strides.getD 0 0 +
          ih.val*result.tensor.strides.getD 1 0 + id.val*result.tensor.strides.getD 2 0)) := by
  obtain ⟨st,oh,od,hs,ha⟩ := reshape_view_addresses region fresh a b h d sa sb sh sd
    result success isView
  simp only [hs,List.getD_cons_zero,List.getD_cons_succ]
  rw [ha ia ib ih id]

#axiomsClean flattened_index_lt
#axiomsClean rank4_view_address
#axiomsClean rank4_view_addresses
#axiomsClean reshape_view_addresses
#axiomsClean reshape_view_read
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KcacheCopyTriton.Host.View
