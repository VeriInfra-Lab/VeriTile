import bench.repairs.tritonbench_g.token_softmax_bloom.Proofs.KernelAdmission

namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom.KernelAddresses
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked
open KernelTensor
noncomputable section

def word (n : Nat) : BitVec 64 := BitVec.ofNat 64 n
def Exact (w : BitVec 64) (n : Nat) := w.toInt=(n : Int)

theorem word_exact (n : Nat) (hn : n  <  2^63) : Exact (word n) n := by
  unfold Exact word
  rw [BitVec.toInt_ofNat']
  apply Int.bmod_eq_of_le <;> norm_num <;> omega

theorem source_offset (h s i rs cs : Nat) (bound : h*rs+(s+i)*cs  <  2^63) :
    Exact (word h*word rs+(word s+word i)*word cs) (h*rs+(s+i)*cs) := by
  simpa only [word,BitVec.ofNat_add_ofNat,BitVec.ofNat_mul_ofNat] using word_exact _ bound

theorem source_mask (i n : Nat) (hi : i  <  2^63) (hn : n  <  2^63) :
    (word i).slt (word n)=decide (i  <  n) := by
  have a := word_exact i hi
  have b := word_exact n hn
  simp only [Exact] at a b
  simp [BitVec.slt,a,b]

theorem formed_bound (c : Call) (t : Observation) (h s i : Nat)
    (hh : h  <  rows c.input) (hs : s  ≤  cols c.input) (hi : i  <  block c.maximum) :
    h*rs t+(s+i)*cs t  ≤  formed c t := by
  have a := Nat.mul_le_mul_right (rs t) (show h  ≤  rows c.input-1 by omega)
  have b := Nat.mul_le_mul_right (cs t) (show s+i  ≤  cols c.input+block c.maximum-1 by omega)
  unfold formed
  omega

theorem source_pointer (base offset bound : Nat) (hi : offset  ≤  bound)
    (room : base+4*bound+4  ≤  2^64) (aligned : base%4=0) :
    (word base+word 4*word offset).toNat=base+4*offset ∧
    (word base+word 4*word offset).toNat%4=0 ∧
    (word base+word 4*word offset).toNat+4  ≤  2^64 := by
  have b : base+4*offset  <  2^64 := by omega
  have eq : (word base+word 4*word offset).toNat=base+4*offset := by
    simp only [word,BitVec.ofNat_add_ofNat,BitVec.ofNat_mul_ofNat,BitVec.toNat_ofNat,Nat.mod_eq_of_lt b]
  exact ⟨eq,by rw [eq];omega,by rw [eq];omega⟩

/-- All formed payload indices, not only active loads/stores, are finite and
aligned. Base is the data_ptr observation checked by Python. -/
theorem checked_payload_addresses (c : Call) (h : MetadataContract c) (launch : active c)
    (head start lane : Nat) (hh : head  <  rows c.input) (hs : start  ≤  cols c.input)
    (hi : lane  <  block c.maximum) :
    ∀ t∈[c.input,c.output],
      Exact (word head*word (rs t)+(word start+word lane)*word (cs t))
        (head*rs t+(start+lane)*cs t) ∧
      (word t.metadata.dataPtr.toNat+word 4*
        (word head*word (rs t)+(word start+word lane)*word (cs t))).toNat=
        t.metadata.dataPtr.toNat+4*(head*rs t+(start+lane)*cs t) ∧
      (t.metadata.dataPtr.toNat+4*(head*rs t+(start+lane)*cs t))%4=0 := by
  rcases h.launch launch with ⟨ba,he,bl,no,st,al,tk,ix,ox,sm,lm,ip,op,sp,lp,sep⟩
  intro t ht
  have small := formed_bound c t head start lane hh hs hi
  have align := al t (by simp only [tensors,List.mem_cons,List.not_mem_nil,or_false] at ht ⊢; tauto)
  have bound : formed c t  <  2^63 := by simp only [List.mem_cons,List.not_mem_nil,or_false] at ht; rcases ht with rfl|rfl <;> assumption
  have room : t.metadata.dataPtr.toNat+4*formed c t+4  ≤  2^64 := by
    simp only [List.mem_cons,List.not_mem_nil,or_false] at ht
    rcases ht with rfl|rfl <;> assumption
  have p := source_pointer t.metadata.dataPtr.toNat _ _ small room align
  refine ⟨source_offset _ _ _ _ _ (lt_of_le_of_lt small bound),?_,?_⟩
  · simpa only [word,BitVec.ofNat_add_ofNat,BitVec.ofNat_mul_ofNat] using p.1
  · simpa only [p.1] using p.2.1

theorem metadata_index (c : Call) (h : MetadataContract c) (launch : active c)
    (b : Nat) (hb : b < rows c.start) :
    ∀ t∈[c.start,c.length],Exact (word b*word (rs t)) (b*rs t) ∧
      (word t.metadata.dataPtr.toNat+word 4*(word b*word (rs t))).toNat=t.metadata.dataPtr.toNat+4*(b*rs t) := by
  rcases h.launch launch with ⟨ba,he,bl,no,st,al,tk,ix,ox,sm,lm,ip,op,sp,lp,sep⟩
  intro t ht
  have small : b*rs t ≤ (rows c.start-1)*rs t := Nat.mul_le_mul_right _ (by omega)
  have bound : (rows c.start-1)*rs t < 2^63 := by
    simp only [List.mem_cons,List.not_mem_nil,or_false] at ht
    rcases ht with rfl|rfl <;> assumption
  have room : t.metadata.dataPtr.toNat+4*((rows c.start-1)*rs t)+4 ≤ 2^64 := by
    simp only [List.mem_cons,List.not_mem_nil,or_false] at ht
    rcases ht with rfl|rfl <;> assumption
  have align := al t (by simp only [tensors,List.mem_cons,List.not_mem_nil,or_false] at ht ⊢;tauto)
  have pointer := source_pointer _ _ _ small room align
  refine ⟨?_,?_⟩
  · simpa only [word,BitVec.ofNat_mul_ofNat] using word_exact _ (lt_of_le_of_lt small bound)
  · simpa only [word,BitVec.ofNat_mul_ofNat] using pointer.1

theorem checked_mask (c : Call) (h : MetadataContract c) (launch : active c)
    (n i : Nat) (hn : n ≤ c.maximum) (hi : i < block c.maximum) :
    (word i).slt (word n)=decide (i < n) := by
  have bound := (h.launch launch).2.2.1
  have cover := VeriTile.Bench.Repairs.SquareMatrix.block_covers c.maximum
  change c.maximum ≤ block c.maximum at cover
  apply source_mask <;> omega


#axiomsClean checked_payload_addresses
#axiomsClean metadata_index
#axiomsClean checked_mask
end
end VeriTile.Bench.Repairs.TokenSoftmaxBloom.KernelAddresses
