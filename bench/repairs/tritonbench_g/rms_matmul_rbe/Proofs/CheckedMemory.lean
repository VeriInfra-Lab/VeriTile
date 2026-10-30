import bench.repairs.tritonbench_g.rms_matmul_rbe.Proofs.CheckedGeometry
import bench.repairs.tritonbench_g.rms_matmul_rbe.Proofs.Index

namespace VeriTile.Bench.Repairs.RmsMatmulRbe.Checked
noncomputable section
set_option maxHeartbeats 5000000
open Contract

/-- Ordinary successful Tensor/storage observations. Strides of size-one
dimensions need not be canonical. Contiguous logical elements start at
data_ptr; span includes the original storage offset and complete backing. -/
structure Ordinary (t : Contract.Tensor) : Prop where
  count64 : t.metadata.numel<2^63
  span : width t.metadata.dtype*(t.offset+t.metadata.numel)≤t.bytes
  addressSpace : t.base+t.bytes≤2^64
  pointer : 0<t.metadata.numel → t.metadata.dataPtr=BitVec.ofNat 64 (t.base+width t.metadata.dtype*t.offset)

theorem width_positive (dtype : VeriTile.Torch.DType) : 0<width dtype := by
  unfold width
  split_ifs <;> decide
theorem Ordinary.positiveBytes (ordinary : Ordinary t) (nonempty : 0<t.metadata.numel) : 0<t.bytes := by
  have hw := width_positive t.metadata.dtype
  have hs := ordinary.span
  nlinarith
theorem disjoint_base_ne (a b : Contract.Tensor) (h : disjoint a b)
    (ha : 0<a.bytes) (hb : 0<b.bytes) : a.base≠b.base := by
  unfold disjoint overlap at h
  omega
theorem disjoint_symm (a b : Contract.Tensor) (h : disjoint a b) : disjoint b a := by
  simpa only [disjoint,overlap,and_comm] using h

def pointer64 (t : Contract.Tensor) (j : Nat) : BitVec 64 :=
  t.metadata.dataPtr+BitVec.ofNat 64 (width t.metadata.dtype)*BitVec.ofNat 64 j
theorem pointer_exact (t : Contract.Tensor) (ordinary : Ordinary t) (j : Nat) (hj : j<t.metadata.numel) :
    (pointer64 t j).toNat=t.base+width t.metadata.dtype*(t.offset+j) ∧
    (BitVec.ofNat 64 j).toInt=(j:Int) ∧
    t.base≤(pointer64 t j).toNat ∧
    (pointer64 t j).toNat+width t.metadata.dtype≤t.base+t.bytes := by
  have hw := width_positive t.metadata.dtype
  have hspan := ordinary.span
  have hspace := ordinary.addressSpace
  have hlast : t.base+width t.metadata.dtype*(t.offset+j)+width t.metadata.dtype≤t.base+t.bytes := by
    nlinarith
  have hb : t.base+width t.metadata.dtype*(t.offset+j)<2^64 := by omega
  have he : (pointer64 t j).toNat=t.base+width t.metadata.dtype*(t.offset+j) := by
    rw [pointer64,ordinary.pointer (by omega),BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
    have rearrange : t.base+width t.metadata.dtype*t.offset+width t.metadata.dtype*j=
        t.base+width t.metadata.dtype*(t.offset+j) := by ring
    rw [rearrange,BitVec.toNat_ofNat,Nat.mod_eq_of_lt hb]
  refine ⟨he,?_,by rw [he]; omega,by rw [he]; exact hlast⟩
  exact DequantizeMatmul.signed64_exact j (hj.trans ordinary.count64)
theorem pointer_aligned (t : Contract.Tensor) (ordinary : Ordinary t) (admitted : TensorAdmitted t types)
    (j : Nat) (hj : j<t.metadata.numel) : (pointer64 t j).toNat%width t.metadata.dtype=0 := by
  have initial := admitted.2.2.2.2.2.2 (by omega)
  have hb : t.base+width t.metadata.dtype*t.offset<2^64 := by
    have hw := width_positive t.metadata.dtype
    have hs := ordinary.span
    have ha := ordinary.addressSpace
    nlinarith
  have pointer0 : t.metadata.dataPtr.toNat=t.base+width t.metadata.dtype*t.offset := by
    rw [ordinary.pointer (by omega),BitVec.toNat_ofNat,Nat.mod_eq_of_lt hb]
  rw [(pointer_exact t ordinary j hj).1]
  rw [pointer0] at initial
  have rearrange : t.base+width t.metadata.dtype*(t.offset+j)=
      (t.base+width t.metadata.dtype*t.offset)+j*width t.metadata.dtype := by ring
  rw [rearrange,Nat.add_mul_mod_self_right]
  exact initial

def PointerValid (t : Contract.Tensor) (j : Nat) : Prop :=
  (pointer64 t j).toNat=t.base+width t.metadata.dtype*(t.offset+j) ∧
  (BitVec.ofNat 64 j).toInt=(j:Int) ∧ t.base≤(pointer64 t j).toNat ∧
  (pointer64 t j).toNat+width t.metadata.dtype≤t.base+t.bytes ∧
  (pointer64 t j).toNat%width t.metadata.dtype=0
theorem pointer_valid (t : Contract.Tensor) (ordinary : Ordinary t) (admitted : TensorAdmitted t types)
    (j : Nat) (hj : j<t.metadata.numel) : PointerValid t j := by
  obtain ⟨he,hi,hlo,hhi⟩ := pointer_exact t ordinary j hj
  exact ⟨he,hi,hlo,hhi,pointer_aligned t ordinary admitted j hj⟩

/-- torch.empty allocates Q, preserves caller storage and returns its ordinary
metadata. Freshness is the allocator contract, not a task output assumption. -/
structure Allocation (i : Inputs) where
  q : Contract.Tensor
  ordinary : ∀ t∈i.reads++[i.k,i.v],Ordinary t
  qOrdinary : Ordinary q
  qShape : q.metadata.shape=[(shape i).B,(shape i).M,(shape i).N]
  qDType : q.metadata.dtype=fp16
  qDevice : q.metadata.device=i.x.metadata.device
  qLayout : q.metadata.layout=.strided
  qContiguous : q.metadata.isContiguous=Bool.true
  qResolved : q.metadata.isNeg=Bool.false
  qTensor : q.tensor=Bool.true
  qOffset : q.offset=0
  qAligned : 0<q.metadata.numel → q.metadata.dataPtr.toNat%2=0
  fresh : ∀ t∈i.reads++[i.k,i.v],disjoint q t

theorem Allocation.qCount (a : Allocation i) : a.q.metadata.numel=(shape i).B*(shape i).M*(shape i).N := by
  simp [VeriTile.Torch.TensorMetadata.numel,a.qShape,Nat.mul_assoc]
theorem Facts.xCount (f : Facts i) : i.x.metadata.numel=(shape i).B*(shape i).M*(shape i).K := by
  simp [VeriTile.Torch.TensorMetadata.numel,f.xShape,Nat.mul_assoc]
theorem Facts.wCount (f : Facts i) (p : Fin 3) : (weight i p).metadata.numel=(shape i).N*(shape i).K := by
  simp [VeriTile.Torch.TensorMetadata.numel,f.weightShape]
theorem Facts.rCount (f : Facts i) : i.rms.metadata.numel=(shape i).K := by
  simp [VeriTile.Torch.TensorMetadata.numel,f.rmsShape]
theorem weight_mem (i : Inputs) (p : Fin 3) : weight i p∈i.reads := by
  fin_cases p <;> simp [weight,Inputs.reads]
theorem output_count (i : Inputs) (h : InputContract i) (a : Allocation i) (p : Fin 3) :
    (output i a.q p).metadata.numel=(shape i).B*(shape i).M*(shape i).N := by
  have f := facts i h
  fin_cases p <;> simp [output,a.qCount,f.kCount,f.vCount]
theorem output_ordinary (a : Allocation i) (p : Fin 3) : Ordinary (output i a.q p) := by
  fin_cases p
  · exact a.qOrdinary
  · exact a.ordinary i.k (by simp)
  · exact a.ordinary i.v (by simp)
theorem output_admitted (i : Inputs) (h : InputContract i) (a : Allocation i) (p : Fin 3) :
    TensorAdmitted (output i a.q p) weightTypes := by
  have f := facts i h
  fin_cases p
  · change TensorAdmitted a.q weightTypes
    refine ⟨a.qTensor,a.qLayout,?_,?_,a.qContiguous,a.qResolved,?_⟩
    · simp [a.qDType,weightTypes]
    · rw [a.qDevice]; exact f.xAdmitted.2.2.2.1
    · intro nonempty
      simpa [a.qDType,width,fp32,fp16] using a.qAligned nonempty
  · exact f.others i.k (by simp [Inputs.others])
  · exact f.others i.v (by simp [Inputs.others])
theorem Allocation.outputBase_injective (h : InputContract i) (a : Allocation i)
    (active : 0<(shape i).B ∧ 0<(shape i).M) :
    Function.Injective (fun p : Fin 3 => (output i a.q p).base) := by
  have f := facts i h
  have hn : 0<(shape i).N := by rw [f.valid.2.2.2.2]; exact Nat.mul_pos f.valid.2.1 f.valid.2.2.1
  have counts : 0<(shape i).B*(shape i).M*(shape i).N := Nat.mul_pos (Nat.mul_pos active.1 active.2) hn
  have hq := a.qOrdinary.positiveBytes (by rw [a.qCount]; exact counts)
  have hk := (a.ordinary i.k (by simp)).positiveBytes (by rw [f.kCount]; exact counts)
  have hv := (a.ordinary i.v (by simp)).positiveBytes (by rw [f.vCount]; exact counts)
  have qk := disjoint_base_ne a.q i.k (a.fresh i.k (by simp)) hq hk
  have qv := disjoint_base_ne a.q i.v (a.fresh i.v (by simp)) hq hv
  have kv := disjoint_base_ne i.k i.v f.kvDisjoint hk hv
  intro p r he
  fin_cases p <;> fin_cases r <;> simp_all [output]

theorem buffers_separate (i : Inputs) (h : InputContract i) (a : Allocation i)
    (active : 0<(shape i).B ∧ 0<(shape i).M) : (buffers i a.q).Separate := by
  have f := facts i h
  have hn : 0<(shape i).N := by rw [f.valid.2.2.2.2]; exact Nat.mul_pos f.valid.2.1 f.valid.2.2.1
  have xPositive : 0 < i.x.metadata.numel := by rw [f.xCount]; exact Nat.mul_pos (Nat.mul_pos active.1 active.2) f.valid.1
  have rPositive : 0 < i.rms.metadata.numel := by rw [f.rCount]; exact f.valid.1
  have wPositive (p) : 0<(weight i p).metadata.numel := by rw [f.wCount]; exact Nat.mul_pos hn f.valid.1
  have oPositive (p) : 0<(output i a.q p).metadata.numel := by
    rw [output_count i h a]; exact Nat.mul_pos (Nat.mul_pos active.1 active.2) hn
  have disj (t) (ht : t∈i.reads) (p : Fin 3) : disjoint t (output i a.q p) := by
    fin_cases p
    · exact disjoint_symm _ _ (a.fresh t (by simp [ht]))
    · exact disjoint_symm _ _ (f.outputsDisjoint t ht).1
    · exact disjoint_symm _ _ (f.outputsDisjoint t ht).2
  have ne (t) (ht : t∈i.reads) (positive : 0<t.metadata.numel) (p : Fin 3) :
      t.base≠(output i a.q p).base := disjoint_base_ne _ _ (disj t ht p)
        ((a.ordinary t (by simp [ht])).positiveBytes positive)
        ((output_ordinary a p).positiveBytes (oPositive p))
  intro p
  exact ⟨ne i.x (by simp [Inputs.reads]) xPositive p,
    ne i.rms (by simp [Inputs.reads]) rPositive p,
    fun w => ne (weight i w) (weight_mem i w) (wPositive w) p⟩

theorem output_injective (i : Inputs) (h : InputContract i) (a : Allocation i) :
    Function.Injective (outputAddress (buffers i a.q) : Cell (shape i) → Address) := by
  intro x y he
  have active : 0<(shape i).B ∧ 0<(shape i).M := ⟨by have := x.2.1.isLt; omega,by have := x.2.2.1.isLt; omega⟩
  have hp : x.1=y.1 := a.outputBase_injective h active (congrArg Prod.fst he)
  have hj := congrArg Prod.snd he
  change ((dense3 (output i a.q x.1) _ _).address _ _ _).2=
    ((dense3 (output i a.q y.1) _ _).address _ _ _).2 at hj
  rw [dense3_address,dense3_address,hp] at hj
  simp only [Prod.snd] at hj
  have hcoords := flat3_injective (Nat.add_left_cancel hj)
  exact Prod.ext hp hcoords

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsMatmulRbe.Checked
