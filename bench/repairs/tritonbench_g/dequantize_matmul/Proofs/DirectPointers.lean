import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.DirectHost
namespace VeriTile.Bench.Repairs.DequantizeMatmul.Direct
noncomputable section
set_option maxHeartbeats 300000

def nativePointer (base : BitVec 64) (bytes offset : Nat) : Nat :=
  (base+BitVec.ofNat 64 (bytes*offset)).toNat

/-- Every active pointer uses the actual data_ptr, existing storage offset,
and canonical contiguous coordinate. Masked-off addresses are not dereferenced. -/
theorem original_pointer (storageBase : Nat → Nat) (t : TensorView bytes)
    (laws : TensorLaws storageBase t) (positive : 0 < bytes)
    (contiguous : t.info.metadata.isContiguous=Bool.true)
    (i : Nat) (hi : i < t.info.metadata.numel) :
    nativePointer t.info.metadata.dataPtr bytes i=storageBase t.info.storage+bytes*t.physicalIndex i := by
  have nonempty : 0 < t.info.metadata.numel := by omega
  have capacity := laws.capacity nonempty
  have lower := Nat.mul_lt_mul_of_pos_left (show t.offset+i < t.offset+t.info.metadata.numel by omega) positive
  have bound : storageBase t.info.storage+bytes*(t.offset+i)<2^64 := by omega
  rw [laws.contiguous contiguous i hi]
  simp only [nativePointer,laws.pointer nonempty,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
  rw [show storageBase t.info.storage+bytes*t.offset+bytes*i=storageBase t.info.storage+bytes*(t.offset+i) by ring]
  exact Nat.mod_eq_of_lt bound

/-- Successful fresh storage provides the corresponding ordinary pointer
window. This generic ABI fact has no source/result arithmetic premise. -/
theorem allocation_pointer (base bytes elements offset : Nat) (positive : 0 < bytes)
    (capacity : base+bytes*elements≤2^64) (active : offset < elements) :
    nativePointer (BitVec.ofNat 64 base) bytes offset=base+bytes*offset := by
  have lower := Nat.mul_lt_mul_of_pos_left active positive
  have bound : base+bytes*offset<2^64 := by omega
  simp only [nativePointer,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
  exact Nat.mod_eq_of_lt bound

theorem fp32_alignment (storageBase : Nat → Nat) (t : TensorView 4)
    (laws : TensorLaws storageBase t) (contiguous : t.info.metadata.isContiguous=Bool.true)
    (alignment : Aligned t.info) (i : Nat) (hi : i < t.info.metadata.numel) :
    nativePointer t.info.metadata.dataPtr 4 i%4=0 := by
  have nonempty : 0 < t.info.metadata.numel := by omega
  have cap := laws.capacity nonempty
  have startBound : storageBase t.info.storage+4*t.offset<2^64 := by omega
  have aligned:=alignment nonempty
  rw [laws.pointer nonempty] at aligned
  simp only [BitVec.toNat_ofNat,Nat.mod_eq_of_lt startBound] at aligned
  rw [original_pointer storageBase t laws (by decide) contiguous i hi,laws.contiguous contiguous i hi]
  simpa [Nat.mul_add,Nat.add_assoc,Nat.add_mod,Nat.mul_mod] using aligned

#axiomsClean original_pointer
#axiomsClean allocation_pointer
#axiomsClean fp32_alignment
end
end VeriTile.Bench.Repairs.DequantizeMatmul.Direct
