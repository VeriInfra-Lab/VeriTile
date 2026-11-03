import bench.tritonbench_g.attention_fwd_triton3.Proofs.Reference

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.SourceSafety
open VeriTile Triton FP.Structural Reference
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
noncomputable section
set_option maxRecDepth 12000
set_option maxHeartbeats 1000000
variable {α : Type} [Inhabited α]

/-- The modeled block pointer converts its signed advance to a natural
offset. Keep the actual signed32 conversion, including wrapping inputs. -/
theorem narrow32_toNat_le (x : Nat) :
    (BitVec.ofInt 32 (x : Int)).toInt.toNat≤x := by
  by_cases small : (x : Int)<2^31
  · have exact : (BitVec.ofInt 32 (x : Int)).toInt=(x : Int) :=
      BitVec.toInt_ofInt_eq_self (by decide) (by omega) small
    rw [exact]
    simp
  · have bound:=BitVec.toInt_lt (x:=BitVec.ofInt 32 (x : Int))
    have bound32 : (BitVec.ofInt 32 (x : Int)).toInt<2147483648 := by norm_num at bound ⊢; exact bound
    omega

theorem lower_nonnegative (c : Call) (a : Inputs α) : 0≤lower c a := by
  unfold lower
  split_ifs <;> omega

/-- Logical loop coordinates and physical K/V offsets can differ when the
signed32 source cast wraps. This definition is independent of execution. -/
def physicalKey (c : Call) (a : Inputs α) (cur : Nat) : Nat :=
  if c.window≠0 ∧ c.complement=0 then
    (BitVec.ofInt 32 (lower c a)).toInt.toNat+(cur-(lower c a).toNat)
  else cur

theorem physical_le (c : Call) (a : Inputs α) (cur : Nat)
    (started : (lower c a).toNat≤cur) : physicalKey c a cur≤cur := by
  have exactLower : ((lower c a).toNat : Int)=lower c a :=
    Int.toNat_of_nonneg (lower_nonnegative c a)
  have narrowed:=narrow32_toNat_le (lower c a).toNat
  rw [exactLower] at narrowed
  unfold physicalKey
  split_ifs <;> omega

theorem physical_advance (c : Call) (a : Inputs α) (cur : Nat)
    (started : (lower c a).toNat≤cur) :
    physicalKey c a (cur+c.BN)=physicalKey c a cur+c.BN := by
  unfold physicalKey
  split_ifs <;> omega

/-- Outside the signed32 range the source conversion is not an identity. -/
theorem signed32_wrap : (BitVec.ofInt 32 (2^31 : Int)).toInt= -(2^31) := by decide

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.SourceSafety
