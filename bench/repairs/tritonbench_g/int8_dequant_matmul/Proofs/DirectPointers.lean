import bench.repairs.tritonbench_g.int8_dequant_matmul.Proofs.DirectHost
namespace VeriTile.Bench.Repairs.Int8DequantMatmul.Direct
noncomputable section

def nativePointer (base : BitVec 64) (bytes offset : Nat) : Nat :=
  (base+BitVec.ofNat 64 (bytes*offset)).toNat

def logicalAddress (t : TensorView) (i : Nat) : Address := (t.info.storage,t.physicalIndex i)

theorem original_pointer (bytes : Nat) (base : Nat → Nat) (t : TensorView)
    (laws : TensorLaws bytes base t) (positive : 0<bytes)
    (contiguous : t.info.metadata.isContiguous=Bool.true) (i : Nat) (hi : i<t.info.metadata.numel) :
    nativePointer t.info.metadata.dataPtr bytes i=base t.info.storage+bytes*t.physicalIndex i := by
  have nonempty : 0<t.info.metadata.numel := by omega
  have cap:=laws.capacity nonempty
  have bound : base t.info.storage+bytes*(t.info.offset+i)<2^64 := by
    have := Nat.mul_lt_mul_of_pos_left (show t.info.offset+i<t.info.offset+t.info.metadata.numel by omega) positive
    omega
  rw [laws.contiguous contiguous i hi]
  simp only [nativePointer,laws.pointer nonempty,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
  rw [show base t.info.storage+bytes*t.info.offset+bytes*i=base t.info.storage+bytes*(t.info.offset+i) by ring]
  exact Nat.mod_eq_of_lt bound

theorem original_read (bytes : Nat) (base : Nat → Nat) (t : TensorView) (laws : TensorLaws bytes base t)
    (contiguous : t.info.metadata.isContiguous=Bool.true) (m : Memory F H)
    (i : Nat) (hi : i<t.info.metadata.numel) :
    m (t.info.storage,t.info.offset+i)=m (logicalAddress t i) := by
  simp only [logicalAddress,laws.contiguous contiguous i hi]

theorem fp32_alignment (base : Nat → Nat) (t : TensorView) (laws : TensorLaws 4 base t)
    (contiguous : t.info.metadata.isContiguous=Bool.true) (aligned : Aligned t.info)
    (i : Nat) (hi : i<t.info.metadata.numel) : nativePointer t.info.metadata.dataPtr 4 i%4=0 := by
  have ne : 0<t.info.metadata.numel := by omega
  have cap:=laws.capacity ne
  have bound : base t.info.storage+4*t.info.offset<2^64 := by omega
  have h:=aligned ne
  rw [laws.pointer ne] at h
  simp only [BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound] at h
  rw [original_pointer 4 base t laws (by decide) contiguous i hi,laws.contiguous contiguous i hi]
  simpa [Nat.mul_add,Nat.add_assoc,Nat.add_mod,Nat.mul_mod] using h

theorem matrix_shape (t : TensorInfo) (rank : t.metadata.shape.length=2) : t.metadata.shape=[rows t,cols t] := by
  rcases t with ⟨⟨dtype,layout,device,shape,contiguous,neg,pointer⟩,storage,offset⟩
  rcases shape with _|⟨r,tail⟩
  · simp at rank
  rcases tail with _|⟨c,tail⟩
  · simp at rank
  cases tail with
  | nil=>rfl
  | cons _ _=>simp at rank

theorem matrix_numel (t : TensorInfo) (rank : t.metadata.shape.length=2) : t.metadata.numel=rows t*cols t := by
  unfold VeriTile.Torch.TensorMetadata.numel
  rw [matrix_shape t rank]
  simp

structure RequestLaws (base : Nat → Nat) (r : Request) : Prop where
  a : TensorLaws 1 base r.a
  b : TensorLaws 1 base r.b
  x : TensorLaws 4 base r.x
  w : TensorLaws 4 base r.w
  bias : ∀ t,r.bias=some t → TensorLaws 4 base t

/-- Every source operand is read from the corresponding original tensor view.
All no-negative-view predicates are checked by InputContract; no input copy. -/
theorem original_operands (base : Nat → Nat) (r : Request) (laws : RequestLaws base r)
    (m : Memory F H) (out row n k : Nat) (hr : row<(requestShape r).M)
    (hn : n<(requestShape r).N) (hk : k<(requestShape r).K) :
    m ((requestBuffers r out).a,(requestBuffers r out).aOffset+row*(requestShape r).K+k)=m (logicalAddress r.a (row*(requestShape r).K+k)) ∧
    m ((requestBuffers r out).b,(requestBuffers r out).bOffset+k*(requestShape r).N+n)=m (logicalAddress r.b (k*(requestShape r).N+n)) ∧
    m ((requestBuffers r out).x,(requestBuffers r out).xOffset+row)=m (logicalAddress r.x row) ∧
    m ((requestBuffers r out).w,(requestBuffers r out).wOffset+n)=m (logicalAddress r.w n) := by
  have ha : row*(requestShape r).K+k<r.a.info.metadata.numel := by
    rw [matrix_numel _ r.contract.rank.1]
    change row*cols r.a.info+k<rows r.a.info*cols r.a.info
    change row<rows r.a.info at hr;change k<cols r.a.info at hk
    nlinarith
  have hb : k*(requestShape r).N+n<r.b.info.metadata.numel := by
    rw [matrix_numel _ r.contract.rank.2.1,r.contract.shape.1]
    change k*cols r.b.info+n<cols r.a.info*cols r.b.info
    change k<cols r.a.info at hk;change n<cols r.b.info at hn
    nlinarith
  have hx : row<r.x.info.metadata.numel := by simpa [VeriTile.Torch.TensorMetadata.numel,r.contract.shape.2.1] using hr
  have hw : n<r.w.info.metadata.numel := by simpa [VeriTile.Torch.TensorMetadata.numel,r.contract.shape.2.2] using hn
  exact ⟨by simpa [requestBuffers,buffers,Nat.add_assoc] using original_read 1 base r.a laws.a r.contract.contiguous.1 m _ ha,
    by simpa [requestBuffers,buffers,Nat.add_assoc] using original_read 1 base r.b laws.b r.contract.contiguous.2.1 m _ hb,
    original_read 4 base r.x laws.x r.contract.contiguous.2.2.1 m row hx,
    original_read 4 base r.w laws.w r.contract.contiguous.2.2.2 m n hw⟩

#axiomsClean original_pointer
#axiomsClean original_read
#axiomsClean fp32_alignment
#axiomsClean original_operands
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.Direct
