import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.SourceSafetySteps
import bench.repairs.tritonbench_g.attention_fwd_triton3.Proofs.Host

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.SourceSafety
open VeriTile Triton FP.Structural Safety FP.ProfiledRewrite
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open Reference
noncomputable section
set_option maxHeartbeats 3000000
variable {α : Type} [Inhabited α]

/-- Only source-free six-buffer address windows; no execution premise. -/
structure Windows (c : Call) (a : Inputs α) (bounds : RegionName → Nat)
    extends PreludeBounds c a bounds : Prop where
  keys : ∀ (k : Fin c.NK) (d : Fin c.D),
    batch c a*c.k.batch+kvhead c a*c.k.head+d.val*c.k.dim+k.val*c.k.row<bounds "K"
  values : ∀ (k : Fin c.NK) (d : Fin c.D),
    batch c a*c.v.batch+kvhead c a*c.v.head+k.val*c.v.row+d.val*c.v.dim<bounds "V"

def keyAt (c : Call) (a : Inputs α) (cur : Nat) : BlockPtr :=
  { kptr c a with offsets := [0,cur] }
def valueAt (c : Call) (a : Inputs α) (cur : Nat) : BlockPtr :=
  { vptr c a with offsets := [cur,0] }

def PointerAt (c : Call) (a : Inputs α) (old : Carry c α) (cur : Nat) : Prop :=
  old.keyPointer=keyAt c a cur ∧ old.valuePointer=valueAt c a cur

theorem lower_nonnegative (c : Call) (a : Inputs α) : 0≤lower c a := by
  unfold lower
  split_ifs <;> omega

theorem lower_nat (c : Call) (a : Inputs α) :
    (lower c a).toNat=RealKeys.lower a.tile c.BM c.BN c.window c.complement c.offset c.size := by
  simp only [lower,RealKeys.lower,WindowBounds.lowerBound]
  split_ifs <;> omega

theorem lower_aligned (c : Call) (a : Inputs α) (positive : 0<c.BN) :
    c.BN ∣ (lower c a).toNat := by
  rw [lower_nat]
  unfold RealKeys.lower
  split_ifs
  · exact WindowBounds.lower_bound_aligned _ _ _ _ _ positive
  · exact dvd_zero _

theorem upper_bound (c : Call) (a : Inputs α) : upper c a≤c.NK := by
  unfold upper
  split_ifs <;> omega

theorem initial_pointer (c : Call) (o : Ops α) (a : Inputs α) :
    PointerAt c a (initial c o a) (lower c a).toNat := by
  unfold PointerAt initial
  by_cases hw : c.window≠0 ∧ c.complement=0
  · simp [hw,keyAt,valueAt,kptr,vptr,BlockPtr.advance_2d_offsets_int]
  · simp [hw,lower,keyAt,valueAt,kptr,vptr]

theorem advance_pointer (c : Call) (o : Ops α) (a : Inputs α)
    (old : Carry c α) (cur : Nat) (pointers : PointerAt c a old cur) :
    PointerAt c a (advance c o a cur old) (cur+c.BN) := by
  rcases pointers with ⟨hk,hv⟩
  simp [PointerAt,advance,hk,hv,keyAt,valueAt,kptr,vptr]

theorem full_block (c : Call) (cur : Nat) (aligned : c.BN∣cur)
    (active : cur<c.NK) (even : c.params.evenN≠0) : cur+c.BN≤c.NK := by
  have total : c.BN∣c.NK := Nat.dvd_of_mod_eq_zero (evenN c even)
  have hd : c.BN∣c.NK-cur := Nat.dvd_sub total aligned
  have hle := Nat.le_of_dvd (show 0<c.NK-cur by omega) hd
  omega

theorem key_bounds (c : Call) (a : Inputs α) (bounds : RegionName → Nat)
    (windows : Windows c a bounds) (cur : Nat) (aligned : c.BN∣cur)
    (active : (cur : Int)<upper c a) :
    BlockBounds bounds (keyAt c a cur) [c.D,c.BN]
      (if c.params.evenN≠0 then [] else [0,1]) := by
  intro i checked
  have nk : cur<c.NK := by have := upper_bound c a; omega
  have bound : cur+i.2.1.val<c.NK := by
    by_cases he : c.params.evenN≠0
    · have whole := full_block c cur aligned nk he
      have := i.2.1.isLt
      omega
    · have h : i.1.val<c.D ∧ cur+i.2.1.val<c.NK := by
        simpa [keyAt,kptr,he] using checked
      exact h.2
  simpa [keyAt,kptr] using windows.keys ⟨cur+i.2.1.val,bound⟩ i.1

theorem value_bounds (c : Call) (a : Inputs α) (bounds : RegionName → Nat)
    (windows : Windows c a bounds) (cur : Nat) (aligned : c.BN∣cur)
    (active : (cur : Int)<upper c a) :
    BlockBounds bounds (valueAt c a cur) [c.BN,c.D]
      (if c.params.evenN≠0 then [] else [0,1]) := by
  intro i checked
  have nk : cur<c.NK := by have := upper_bound c a; omega
  have bound : cur+i.1.val<c.NK := by
    by_cases he : c.params.evenN≠0
    · have whole := full_block c cur aligned nk he
      have := i.1.isLt
      omega
    · have h : cur+i.1.val<c.NK ∧ i.2.1.val<c.D := by
        simpa [valueAt,vptr,he] using checked
      exact h.1
  simpa [valueAt,vptr] using windows.values ⟨cur+i.1.val,bound⟩ i.2.1

#axiomsClean key_bounds
#axiomsClean value_bounds
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.SourceSafety
