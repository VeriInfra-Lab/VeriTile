import bench.repairs.tritonbench_g.embedding_triton_kernel.Proofs.Host
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel.TensorBinding
open VeriTile.Triton Physical Host
open VeriTile.Torch.FP32
noncomputable section

/-- Int32 uses the same ordinary storage/contiguity observations as FP32;
only its signed interpretation differs. No embedding result occurs here. -/
structure Int32Laws (t : Tensor) : Prop where
  dtype : t.metadata.dtype = .other "int32"
  layout : t.metadata.layout = .strided
  inBounds : ∀ i,i<t.metadata.numel → t.physicalIndex i<t.storage.elements
  pointer : 0<t.metadata.numel → t.dataPtr=BitVec.ofNat 64 (t.storage.address t.storageOffset)
  contiguous : t.metadata.isContiguous=Bool.true →
    ∀ i,i<t.metadata.numel → t.physicalIndex i=t.storageOffset+i

structure Objects (c : Request) where
  (ids weight output : Tensor)
  idsMetadata : ids.metadata=c.ids
  weightMetadata : weight.metadata=c.weight
  outputMetadata : output.metadata=c.output
  idsLaws : Int32Laws ids
  weightLaws : TensorLaws weight
  outputLaws : TensorLaws output

def fpRead (m : Memory) (byte : Nat) : VeriTile.Torch.FP32.Word := ⟨m byte⟩

def weightValue (c : Request) (objects : Objects c) (m : Memory) (r d : Nat) : Physical.Word :=
  (logicalValue (fpRead m) objects.weight objects.weightLaws (r*c.hidden+d)).bits

def outputValue (c : Request) (objects : Objects c) (m : Memory) (r d : Nat) : Physical.Word :=
  (logicalValue (fpRead m) objects.output objects.outputLaws (r*c.hidden+d)).bits

def indexValue (t : Tensor) (_laws : Int32Laws t) (m : Memory) (i : Nat) : Int :=
  let raw := m (t.storage.address (t.physicalIndex i))
  if t.metadata.isNeg then (-raw).toInt else raw.toInt

theorem fp_base (t : Tensor) (laws : TensorLaws t) (hc : t.metadata.isContiguous=Bool.true)
    (hp : 0<t.metadata.numel) : (dataView t laws hc).base=t.metadata.dataPtr.toNat := by
  have ptr := actual_pointer t laws hc hp
  have span := (dataView t laws hc).spanFits
  have bound : (dataView t laws hc).base<2^64 := by
    change (dataView t laws hc).base+4*t.metadata.numel≤2^64 at span
    omega
  have eq := congrArg BitVec.toNat ptr
  simp only [DataView.address,Nat.mul_zero,Nat.add_zero,Tensor.dataPtr,BitVec.toNat_ofNat] at eq
  rw [Nat.mod_eq_of_lt bound] at eq
  exact eq

theorem fp_direct (t : Tensor) (laws : TensorLaws t)
    (hc : t.metadata.isContiguous=Bool.true) (hn : t.metadata.isNeg=Bool.false)
    (m : Memory) (i : Nat) (hi : i<t.metadata.numel) :
    m (address t.metadata i)=(logicalValue (fpRead m) t laws i).bits := by
  have same := congrArg (fun x : VeriTile.Torch.FP32.Word=>x.bits)
    (direct_read_eq_logical (fpRead m) t laws hc hn i hi)
  simpa only [fpRead,DataView.address,fp_base t laws hc (by omega)] using same

theorem int_capacity (t : Tensor) (laws : Int32Laws t)
    (hc : t.metadata.isContiguous=Bool.true) (hp : 0<t.metadata.numel) :
    t.storageOffset+t.metadata.numel≤t.storage.elements := by
  have last := laws.inBounds (t.metadata.numel-1) (by omega)
  rw [laws.contiguous hc _ (by omega)] at last
  omega

theorem int_base (t : Tensor) (laws : Int32Laws t)
    (hc : t.metadata.isContiguous=Bool.true) (hp : 0<t.metadata.numel) :
    t.storage.address t.storageOffset=t.metadata.dataPtr.toNat := by
  have cap := int_capacity t laws hc hp
  have span := t.storage.spanFits
  have bound : t.storage.address t.storageOffset<2^64 := by
    unfold Storage.address
    omega
  have eq := congrArg BitVec.toNat (laws.pointer hp)
  simp only [Tensor.dataPtr,BitVec.toNat_ofNat] at eq
  rw [Nat.mod_eq_of_lt bound] at eq
  exact eq.symm

theorem int_direct (t : Tensor) (laws : Int32Laws t)
    (hc : t.metadata.isContiguous=Bool.true) (hn : t.metadata.isNeg=Bool.false)
    (m : Memory) (i : Nat) (hi : i<t.metadata.numel) :
    (m (address t.metadata i)).toInt=indexValue t laws m i := by
  simp only [indexValue,hn,Bool.false_eq_true,↓reduceIte,laws.contiguous hc i hi]
  have ptr := int_base t laws hc (by omega)
  have addr : address t.metadata i=t.storage.address (t.storageOffset+i) := by
    unfold address Storage.address at *
    omega
  rw [addr]

/-- Empty windows do not access storage; their arbitrary observed pointer is
already a 64-bit word and imposes no unused storage-offset restriction. -/
theorem fp_span (t : Tensor) (laws : TensorLaws t) (hc : t.metadata.isContiguous=Bool.true) :
    t.metadata.dataPtr.toNat+4*t.metadata.numel≤2^64 := by
  by_cases hp : 0<t.metadata.numel
  · have span := (dataView t laws hc).spanFits
    change (dataView t laws hc).base+4*t.metadata.numel≤2^64 at span
    rw [fp_base t laws hc hp] at span
    exact span
  · have ptr := t.metadata.dataPtr.isLt
    simp only [show t.metadata.numel=0 by omega,Nat.mul_zero,Nat.add_zero]
    omega

theorem int_span (t : Tensor) (laws : Int32Laws t) (hc : t.metadata.isContiguous=Bool.true) :
    t.metadata.dataPtr.toNat+4*t.metadata.numel≤2^64 := by
  by_cases hp : 0<t.metadata.numel
  · have cap := int_capacity t laws hc hp
    have span := t.storage.spanFits
    have ptr := int_base t laws hc hp
    unfold Storage.address at ptr
    omega
  · have ptr := t.metadata.dataPtr.isLt
    simp only [show t.metadata.numel=0 by omega,Nat.mul_zero,Nat.add_zero]
    omega

def storage (c : Request) (h : InputContract c) (objects : Objects c) : Native.TensorStorage c := by
  obtain ⟨ci,cw,co,_⟩ := h.2.2.2.1
  refine ⟨?_,?_,?_⟩
  · simpa [objects.idsMetadata] using int_span objects.ids objects.idsLaws (by simpa [objects.idsMetadata] using ci)
  · simpa [objects.weightMetadata] using fp_span objects.weight objects.weightLaws (by simpa [objects.weightMetadata] using cw)
  · simpa [objects.outputMetadata] using fp_span objects.output objects.outputLaws (by simpa [objects.outputMetadata] using co)

def reference (c : Request) (objects : Objects c) (m : Memory) (r d : Nat) : Physical.Word :=
  target 0 c.first c.last (indexValue objects.ids objects.idsLaws m)
    (weightValue c objects m) r d

theorem desired_logical (c : Request) (h : InputContract c) (objects : Objects c)
    (m : Memory) (r d : Nat) (hr : r<c.n) (hd : d<c.hidden) :
    desired c m r d=reference c objects m r d := by
  obtain ⟨ci,cw,co,ni,nw,no⟩ := h.2.2.2.1
  obtain ⟨sn,sw,so⟩ := Bounds.sizes c h
  have ids := int_direct objects.ids objects.idsLaws
    (by simpa [objects.idsMetadata] using ci) (by simpa [objects.idsMetadata] using ni)
    m r (by simpa [objects.idsMetadata,sn] using hr)
  have hid : idAt c m r=indexValue objects.ids objects.idsLaws m r := by
    simpa [idAt,objects.idsMetadata] using ids
  unfold desired reference target
  rw [←hid]
  split_ifs with valid
  · obtain ⟨lo,hi⟩ := id_bounds c m r
    have ri := Bounds.selected_row c h (by omega) _ lo hi valid
    have bound : (idAt c m r-c.first).toNat*c.hidden+d<objects.weight.metadata.numel := by
      rw [objects.weightMetadata,sw]
      exact Bounds.flat_bound _ _ _ _ ri hd
    simpa only [objects.weightMetadata,weightValue] using fp_direct objects.weight objects.weightLaws
      (by simpa [objects.weightMetadata] using cw) (by simpa [objects.weightMetadata] using nw)
      m _ bound
  · rfl

theorem output_logical (c : Request) (h : InputContract c) (objects : Objects c)
    (m : Memory) (r d : Nat) (hr : r<c.n) (hd : d<c.hidden) :
    m (address c.output (r*c.hidden+d))=outputValue c objects m r d := by
  obtain ⟨_,_,co,_,_,no⟩ := h.2.2.2.1
  have bound : r*c.hidden+d<objects.output.metadata.numel := by
    rw [objects.outputMetadata,(Bounds.sizes c h).2.2]
    exact Bounds.flat_bound _ _ _ _ hr hd
  simpa only [objects.outputMetadata,outputValue] using fp_direct objects.output objects.outputLaws
    (by simpa [objects.outputMetadata] using co) (by simpa [objects.outputMetadata] using no) m _ bound

/-- Source output readback through the same caller Tensor objects. -/
theorem returned_tensor (c : Request) (h : InputContract c) (objects : Objects c)
    (m : Memory) (incoming : VeriTile.Torch.Device) :
    (∀ r,r<c.n → ∀ d,d<c.hidden →
      outputValue c objects (Host.run c m objects.output.object incoming).memory r d=
        reference c objects m r d) ∧
    (Host.run c m objects.output.object incoming).value=none ∧
    (Host.run c m objects.output.object incoming).outputObject=objects.output.object := by
  refine ⟨?_,rfl,rfl⟩
  intro r hr d hd
  rw [←output_logical c h objects _ r d hr hd]
  rw [(Host.returned_correct c h (storage c h objects) m objects.output.object incoming).2.2.2.2.1 r hr d hd]
  exact desired_logical c h objects m r d hr hd

def interpretedReference {α : Type} (zero : α) (interpret : Physical.Word → α)
    (c : Request) (objects : Objects c) (m : Memory) (r d : Nat) : α :=
  target zero c.first c.last (indexValue objects.ids objects.idsLaws m)
    (fun i j=>interpret (weightValue c objects m i j)) r d

theorem interpret_reference {α : Type} (zero : α) (interpret : Physical.Word → α)
    (hz : interpret 0=zero) (c : Request) (objects : Objects c) (m : Memory) (r d : Nat) :
    interpret (reference c objects m r d)=interpretedReference zero interpret c objects m r d := by
  unfold reference interpretedReference target
  split_ifs <;> simp_all

#axiomsClean returned_tensor
end
end VeriTile.Bench.Repairs.EmbeddingTritonKernel.TensorBinding
