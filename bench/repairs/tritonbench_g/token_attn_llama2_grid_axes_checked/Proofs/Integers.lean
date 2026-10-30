import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.Domain
namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.Whole
open VeriTile.Triton FP.Structural VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue
noncomputable section
set_option maxHeartbeats 4000000

/-- Every actually formed padded-lane expression fits signed int32. Q/token
arithmetic precedes the 0/1 loop; Loc/K/Out arithmetic occurs only in its body. -/
def Formed (c : Call) (m : Metadata c) : Prop :=
  ∀ b<c.batches, ∀ h<c.heads, ∀ n<(c.maxLength+31)/32, ∀ i<32, ∀ d<c.dim,
    n*32+i<2147483648 ∧
    b*c.qBatch+h*c.qHead+d*c.qDim<2147483648 ∧
    (n*32<m.len b →
      c.maxLength-m.len b+(n*32+i)<2147483648 ∧
      c.locBatch*b+c.locSeq*(c.maxLength-m.len b+(n*32+i))<2147483648 ∧
      (if n*32+i<m.len b then m.page b (n*32+i) else 0)*c.kBatch+
        (h/(c.heads/c.kvHeads))*c.kHead+d*c.kDim<2147483648 ∧
      h*c.outHead+(m.begin b+(n*32+i))*c.outSeq<2147483648)

theorem formed (c : Call) (m : Metadata c) (v : Valid c m) : Formed c m := by
  intro b hb h hh n hn i hi d hd
  have env := v.signedEnvelope
  have hl := length_bound c m v b hb
  have hs := (v.starts ⟨b,hb⟩).2
  dsimp at hs
  have ht : n*32+i<c.maxLength+32 := by omega
  have qb := Nat.mul_le_mul_right c.qBatch (Nat.le_of_lt hb)
  have qh := Nat.mul_le_mul_right c.qHead (Nat.le_of_lt hh)
  have qd := Nat.mul_le_mul_right c.qDim (Nat.le_of_lt hd)
  refine ⟨by omega,by omega,?_⟩
  intro block
  have window : c.maxLength-m.len b+(n*32+i)<c.maxLength+32 := by omega
  have locb := Nat.mul_le_mul_left c.locBatch (Nat.le_of_lt hb)
  have locw := Nat.mul_le_mul_left c.locSeq (Nat.le_of_lt window)
  have pg : (if n*32+i<m.len b then m.page b (n*32+i) else 0)≤c.cacheRows := by
    split_ifs with live
    · exact Nat.le_of_lt (v.pages ⟨b,hb⟩ _ live).2
    · omega
  have kp := Nat.mul_le_mul_right c.kBatch pg
  have kh := Nat.mul_le_mul_right c.kHead (Nat.le_of_lt (grouped_head c m v h hh))
  have kd := Nat.mul_le_mul_right c.kDim (Nat.le_of_lt hd)
  have oh := Nat.mul_le_mul_right c.outHead (Nat.le_of_lt hh)
  have ot : m.begin b+(n*32+i)<c.outputTokens+32 := by omega
  have oo := Nat.mul_le_mul_right c.outSeq (Nat.le_of_lt ot)
  exact ⟨by omega,by omega,by omega,by omega⟩

/-- Metadata casts and the signed source subtraction retain their actual data. -/
theorem metadata_range (c : Call) (m : Metadata c) (v : Valid c m)
    (b : Fin c.batches) :
    signed32 (m.lengths b)=m.lengths b ∧ signed32 (m.starts b)=m.starts b ∧
    ((c.maxLength-m.len b.val : Nat) : Int)=(c.maxLength : Int)-m.lengths b := by
  have env := v.signedEnvelope
  have len := (signed_length c m v b).1
  have hs := (v.starts b).2
  have start : (m.begin b.val : Int)=m.starts b := by simp [Metadata.begin,Int.toNat_of_nonneg (v.starts b).1]
  have hl := length_bound c m v b.val b.isLt
  exact ⟨signed32_exact _ (v.lengths b).1 (by omega),
    signed32_exact _ (v.starts b).1 (by omega),(signed_length c m v b).2⟩

#axiomsClean formed
#axiomsClean metadata_range
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.Whole
