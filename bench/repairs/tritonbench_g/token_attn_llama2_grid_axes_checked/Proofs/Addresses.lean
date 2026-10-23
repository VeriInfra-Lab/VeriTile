import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Contract

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked
open VeriTile.Triton FP.Structural Whole
noncomputable section

def index32 (n : Nat) : BitVec 32 := BitVec.ofNat 32 n
def pointer64 (base offset : Nat) : BitVec 64 :=
  BitVec.ofNat 64 base+BitVec.ofNat 64 4*BitVec.ofNat 64 offset

theorem index32_exact (n : Nat) (bound : n<2^31) : (index32 n).toInt=(n : Int) := by
  unfold index32
  rw [BitVec.toInt_ofNat']
  apply Int.bmod_eq_of_le <;> norm_num <;> omega

theorem mask32_exact (a b : Nat) (ha : a<2^31) (hb : b<2^31) :
    (index32 a).slt (index32 b)=decide (a<b) := by
  simp [BitVec.slt,index32_exact a ha,index32_exact b hb]

/-- Pointer arithmetic always uses uint64 modular arithmetic, including
masked lanes. Only actual accesses require the nonwrapping view bound. -/
theorem pointer_mod (base offset : Nat) :
    (pointer64 base offset).toNat=(base+4*offset)%2^64 := by
  simp only [pointer64,BitVec.ofNat_add_ofNat,BitVec.ofNat_mul_ofNat,BitVec.toNat_ofNat]

theorem pointer_exact (t : TensorCall) (words : Nat → Word) (contract : InputContract t words)
    (launch : active t) (o : Observation) (member : o∈t.observations)
    (i : Nat) (hi : i<o.metadata.numel) :
    (pointer64 o.metadata.dataPtr.toNat i).toNat=o.metadata.dataPtr.toNat+4*i ∧
    o.metadata.dataPtr.toNat+4*i+4≤2^64 ∧
    (pointer64 o.metadata.dataPtr.toNat i).toNat%4=0 := by
  have facts := (contract.2.2.2 launch).2.1.2.1 o member
  have bound : o.metadata.dataPtr.toNat+4*i<2^64 := by omega
  rw [pointer_mod,Nat.mod_eq_of_lt bound]
  have alignment := facts.2 (by omega)
  exact ⟨rfl,by omega,by omega⟩

theorem metadata_address (t : MetadataTensor) (contiguous : t.metadata.isContiguous=Bool.true)
    (i : Nat) (hi : i<t.metadata.numel) :
    t.metadata.dataPtr.toNat+4*i=t.storage.address (t.storageOffset+i) := by
  have positive : 0<t.metadata.numel := by omega
  have first := t.bounds contiguous 0 positive
  have span := t.storage.spanFits
  have base : t.storage.address t.storageOffset<2^64 := by
    simp only [VeriTile.Torch.FP32.Storage.address] at *
    omega
  rw [t.pointer positive]
  change (t.storage.address t.storageOffset)%2^64+4*i=_
  rw [Nat.mod_eq_of_lt base]
  simp [VeriTile.Torch.FP32.Storage.address,Nat.mul_add,Nat.add_assoc]

/-- Every source-formed signed32 index, including inactive padded lanes, is
bounded by the original envelope; the new axis check does not replace it. -/
theorem formed_indices (t : TensorCall) (words : Nat → Word) (contract : InputContract t words)
    (launch : active t) : Formed t.call (metadata t words) :=
  formed t.call (metadata t words) (valid t words contract launch)

theorem source_length (t : TensorCall) (words : Nat → Word) (b : Nat) (hb : b<t.call.batches) :
    (metadata t words).len b=(words (t.lengths.metadata.dataPtr.toNat+4*b)).toInt.toNat := by
  simp [Whole.Metadata.len,finiteRead,hb,metadata]

theorem source_start (t : TensorCall) (words : Nat → Word) (b : Nat) (hb : b<t.call.batches) :
    (metadata t words).begin b=(words (t.starts.metadata.dataPtr.toNat+4*b)).toInt.toNat := by
  simp [Whole.Metadata.begin,finiteRead,hb,metadata]

theorem native_length_load (t : TensorCall) (words : Nat → Word) (contract : InputContract t words)
    (launch : active t) (b : Nat) (hb : b<t.call.batches) :
    (words (pointer64 t.lengths.metadata.dataPtr.toNat b).toNat).toInt=
      ((metadata t words).len b : Int) := by
  have size : b<t.lengths.metadata.numel := by
    simp [VeriTile.Torch.TensorMetadata.numel,contract.1.lengths,hb]
  have ptr := (pointer_exact t words contract launch t.lengths.observation
    (by simp [TensorCall.observations]) b size).1
  change (pointer64 t.lengths.metadata.dataPtr.toNat b).toNat=_ at ptr
  rw [ptr,source_length t words b hb]
  exact (Int.toNat_of_nonneg (contract.2.2.1.lengths ⟨b,hb⟩).1).symm

theorem native_start_load (t : TensorCall) (words : Nat → Word) (contract : InputContract t words)
    (launch : active t) (b : Nat) (hb : b<t.call.batches) :
    (words (pointer64 t.starts.metadata.dataPtr.toNat b).toNat).toInt=
      ((metadata t words).begin b : Int) := by
  have size : b<t.starts.metadata.numel := by
    simp [VeriTile.Torch.TensorMetadata.numel,contract.1.starts,hb]
  have ptr := (pointer_exact t words contract launch t.starts.observation
    (by simp [TensorCall.observations]) b size).1
  change (pointer64 t.starts.metadata.dataPtr.toNat b).toNat=_ at ptr
  rw [ptr,source_start t words b hb]
  exact (Int.toNat_of_nonneg (contract.2.2.1.starts ⟨b,hb⟩).1).symm

/-- The active suffix gather decodes the actual signed int32 location word.
Inactive prefix values are not restricted by this theorem or the contract. -/
theorem native_page_load (t : TensorCall) (words : Nat → Word) (contract : InputContract t words)
    (launch : active t) (b n : Nat) (hb : b<t.call.batches)
    (hn : n<(metadata t words).len b) :
    (words (pointer64 t.locations.metadata.dataPtr.toNat
      (t.call.locBatch*b+t.call.locSeq*(t.maximum-(metadata t words).len b+n))).toNat).toInt=
      ((metadata t words).page b n : Int) := by
  have domain := valid t words contract launch
  have lengthBound := Whole.length_bound t.call (metadata t words) domain b hb
  change (metadata t words).len b≤t.maximum at lengthBound
  have size := domain.locationBounds b hb
    (t.maximum-(metadata t words).len b+n) (by change _<t.maximum; omega)
  have ptr := (pointer_exact t words contract launch t.locations.observation
    (by simp [TensorCall.observations]) _ size).1
  change (pointer64 t.locations.metadata.dataPtr.toNat _).toNat=_ at ptr
  rw [ptr]
  have exactWord := (Whole.metadata_signed_exact t.call (metadata t words) domain ⟨b,hb⟩ n hn).2.2
  change ((metadata t words).page b n : Int)=finiteRead (metadata t words).locations
    (t.call.locBatch*b+t.call.locSeq*(t.maximum-(metadata t words).len b+n)) at exactWord
  rw [finiteRead,dif_pos size] at exactWord
  exact exactWord.symm

#axiomsClean index32_exact
#axiomsClean mask32_exact
#axiomsClean pointer_mod
#axiomsClean pointer_exact
#axiomsClean metadata_address
#axiomsClean formed_indices
#axiomsClean native_length_load
#axiomsClean native_start_load
#axiomsClean native_page_load
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked
