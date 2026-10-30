import bench.tritonbench_g.int4_matmul.Contract

namespace VeriTile.Bench.Repairs.Int4Matmul.Direct
open Host HelperHost
set_option maxHeartbeats 2000000

theorem tensorContract_iff (t : TensorInfo) (d : DType) (packed : Bool) : TensorContract t d packed ↔
    t.strided=Bool.true ∧ t.descriptor.dtype=d ∧ t.cuda=Bool.true ∧ t.rank=2 ∧
    (0<t.descriptor.rows ∧ 0<t.descriptor.columns) ∧ t.descriptor.offset=0 ∧
    (t.descriptor.lazyConj=Bool.false ∧ t.descriptor.lazyNeg=Bool.false ∧ t.descriptor.requiresGrad=Bool.false) ∧
    t.dataPtr.toNat%elementBytes t.descriptor.dtype=0 ∧
    (if packed then PackedStrides t.descriptor else CanonicalStrides t.descriptor) := by
  constructor
  · intro h;exact ⟨h.layout,h.dtype,h.device,h.rank,h.positive,h.offset,h.flags,h.alignment,h.strides⟩
  · rintro ⟨a,b,c,d,e,f,g,h,i⟩;exact ⟨a,b,c,d,e,f,g,h,i⟩
theorem checkTensor_iff (t : TensorInfo) (d : DType) (packed : Bool) : checkTensor t d packed=.ok () ↔ TensorContract t d packed := by
  rw [tensorContract_iff]
  cases packed <;> simp [checkTensor,PackedStrides,Except.bind,Except.pure,bind,pure,throw,ite_eq_iff,and_assoc,Nat.pos_iff_ne_zero] <;> tauto

theorem packedContract_iff (w s z : TensorInfo) : PackedContract w s z ↔
    TensorContract w .int32 Bool.true ∧ TensorContract s .fp32 Bool.true ∧ TensorContract z .int32 Bool.true ∧
    (packedN w%8=0 ∧ packedK w%s.descriptor.rows=0 ∧ s.descriptor.columns=packedN w ∧
      z.descriptor.rows=s.descriptor.rows ∧ z.descriptor.columns=packedN w/8) ∧
    SameDevice [w,s,z] ∧ Independent [w,s,z] ∧
    (packedK w*packedN w<2^31 ∧ s.descriptor.rows*s.descriptor.columns<2^31) := by
  constructor
  · intro h;exact ⟨h.weight,h.scales,h.zeros,h.geometry,h.device,h.independent,h.bounds⟩
  · rintro ⟨a,b,c,d,e,f,g⟩;exact ⟨a,b,c,d,e,f,g⟩

theorem bind_ok_iff (x : Except InputError Unit) (y : Except InputError Unit) :
    (x >>= fun _=>y)=.ok () ↔ x=.ok () ∧ y=.ok () := by
  cases x with
  | error e=>simp [bind,Except.bind]
  | ok v=>cases v;simp [bind,Except.bind]

theorem checkPacked_iff (w s z : TensorInfo) : checkPacked w s z=.ok () ↔ PackedContract w s z := by
  rw [packedContract_iff]
  simp [checkPacked,bind_ok_iff,checkTensor_iff,Except.pure,pure,throw,ite_eq_iff,and_assoc]
  tauto

theorem outputContract_iff (input weight out : TensorInfo) : OutputContract input weight out ↔
    TensorContract out .fp32 Bool.false ∧ out.descriptor.rows=input.descriptor.rows ∧ out.descriptor.columns=packedN weight := by
  constructor
  · intro h;exact ⟨h.tensor,h.shape⟩
  · rintro ⟨a,b,c⟩;exact ⟨a,b,c⟩
theorem checkOutput_iff (input weight out : TensorInfo) : checkOutput input weight out=.ok () ↔ OutputContract input weight out := by
  rw [outputContract_iff]
  simp [checkOutput,bind_ok_iff,checkTensor_iff,Except.pure,pure,throw,ite_eq_iff,and_assoc]

theorem matmulContract_iff (a w s z : TensorInfo) (g : Int) (out : Option TensorInfo) : MatmulContract a w s z g out ↔
    PackedContract w s z ∧ TensorContract a .fp32 Bool.false ∧ (0<g ∧ g.toNat=packedGroup w s) ∧
    a.descriptor.columns=packedK w ∧ (∀ t,out=some t → OutputContract a w t) ∧
    SameDevice (matrixInputs a w s z out) ∧ Independent (matrixInputs a w s z out) ∧
    (∀ n∈formed (matrixShape a w s) w,n<2^31) := by
  constructor
  · intro h;exact ⟨h.packed,h.input,h.group,h.inner,h.output,h.device,h.independent,h.bounds⟩
  · rintro ⟨a,b,c,d,e,f,g,h⟩;exact ⟨a,b,c,d,e,f,g,h⟩
theorem checkMatmul_iff (a w s z : TensorInfo) (g : Int) (out : Option TensorInfo) :
    checkMatmul a w s z g out=.ok () ↔ MatmulContract a w s z g out := by
  rw [matmulContract_iff]
  cases out <;> simp [checkMatmul,bind_ok_iff,checkPacked_iff,checkTensor_iff,checkOutput_iff,
    Except.pure,pure,throw,ite_eq_iff,and_assoc] <;> tauto

theorem quantMetadataContract_iff (w : TensorInfo) (g r : Int) : QuantMetadataContract w g r ↔
    TensorContract w .fp32 Bool.false ∧ (0<g ∧ w.descriptor.rows%g.toNat=0) ∧
    (w.descriptor.rows%8=0 ∧ w.descriptor.columns%8=0) ∧ r=(w.descriptor.device:Int) ∧
    w.descriptor.rows*w.descriptor.columns<2^31 := by
  constructor
  · intro h;exact ⟨h.weight,h.group,h.packed,h.rank,h.bounds⟩
  · rintro ⟨a,b,c,d,e⟩;exact ⟨a,b,c,d,e⟩
theorem checkQuantMetadata_iff (w : TensorInfo) (g r : Int) :
    checkQuantMetadata w g r=.ok () ↔ QuantMetadataContract w g r := by
  rw [quantMetadataContract_iff]
  simp [checkQuantMetadata,bind_ok_iff,checkTensor_iff,Except.pure,pure,throw,ite_eq_iff,and_assoc]

theorem checkMatmulArguments_iff (a w s z : Argument) (g : ScalarArgument) (out : OutputArgument) :
    checkMatmulArguments a w s z g out=.ok () ↔ AdmittedMatmul a w s z g out := by
  cases a <;> cases w <;> cases s <;> cases z <;> cases g <;> cases out <;>
    simp [checkMatmulArguments,AdmittedMatmul,checkMatmul_iff]

#axiomsClean checkTensor_iff
#axiomsClean checkPacked_iff
#axiomsClean checkMatmul_iff
#axiomsClean checkQuantMetadata_iff
#axiomsClean checkMatmulArguments_iff
end VeriTile.Bench.Repairs.Int4Matmul.Direct
