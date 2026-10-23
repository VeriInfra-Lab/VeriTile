import bench.repairs.tritonbench_g.rms_norm_triton.Contract
import bench.tritonbench_g.rms_norm_triton.Proofs.KernelIndex

namespace VeriTile.Bench.Repairs.RmsNormTriton
open VeriTile.Torch VeriTile.Bench.TritonBenchG.RmsNormTriton
noncomputable section
set_option maxHeartbeats 4000000

theorem checkTensorInput_iff (x w : TensorInfo) (n : NormalizedInfo) (eps : EpsilonInfo) :
    checkTensorInput x w n eps=.ok () ↔ InputContract x w n eps := by
  have fields : InputContract x w n eps ↔
      (x.layout=.strided ∧ x.dtype=.floating .fp32 ∧ x.device.kind="cuda") ∧
      (w.layout=.strided ∧ w.dtype=.floating .fp32 ∧ w.device.kind="cuda") ∧
      x.device=w.device ∧ n.valid=Bool.true ∧ n.dimensions.length≤x.shape.length ∧
      x.shape.drop (dim x n.dimensions)=n.dimensions ∧ w.shape=n.dimensions ∧
      0<columns n.dimensions ∧
      (x.isNeg=Bool.false ∧ x.isConj=Bool.false ∧ w.isNeg=Bool.false ∧ w.isConj=Bool.false) ∧
      eps.pythonFloat=Bool.true ∧ eps.finite=Bool.true ∧ 0<eps.value ∧
      rows x n.dimensions<2^31 ∧ columns n.dimensions<2^31 ∧
      block (columns n.dimensions)<2^31 ∧ rowMax x n.dimensions<2^31 := by
    constructor
    · intro h
      exact ⟨⟨h.input.layout,h.input.dtype,h.input.cuda⟩,⟨h.weight.layout,h.weight.dtype,h.weight.cuda⟩,
        h.device,h.normalizedType,h.rank,h.suffix,h.weightShape,h.positive,h.flags,
        h.epsilonType,h.epsilonFinite,h.epsilonPositive,h.rowsBound,h.columnsBound,h.blockBound,h.rowProducts⟩
    · rintro ⟨⟨a,b,c⟩,⟨d,e,f⟩,g,h,i,j,k,l,m,o,p,q,r,s,t,u⟩
      exact ⟨⟨a,b,c⟩,⟨d,e,f⟩,g,h,i,j,k,l,m,o,p,q,r,s,t,u⟩
  rw [fields]
  simp [checkTensorInput,TensorMetadata.isCuda,ite_eq_iff,Nat.pos_iff_ne_zero]
  tauto

theorem checkInput_iff (x w : Argument) (n : NormalizedInfo) (eps : EpsilonInfo) :
    checkInput x w n eps=.ok () ↔ Admitted x w n eps := by
  cases x <;> cases w <;> simp [checkInput,Admitted,checkTensorInput_iff]

theorem checkPointers_iff (x : TensorInfo) (n : List Nat) (p : Pointers) :
    checkPointers x n p=.ok () ↔ PointerContract x n p := by
  have fields : PointerContract x n p ↔
      (p.input.toNat%4=0 ∧ p.weight.toNat%4=0 ∧ p.output.toNat%4=0) ∧
      p.input.toNat+4*paddedMax x n<2^64 ∧ p.weight.toNat+4*(block (columns n)-1)<2^64 ∧
      p.output.toNat+4*paddedMax x n<2^64 := by
    constructor
    · intro h; exact ⟨h.aligned,h.inputEnd,h.weightEnd,h.outputEnd⟩
    · rintro ⟨a,b,c,d⟩; exact ⟨a,b,c,d⟩
  rw [fields]
  simp [checkPointers,ite_eq_iff]
  tauto

structure Observes (info : TensorInfo) (tensor : Whole.Tensor) : Prop where
  shape : tensor.shape=info.shape
  dtype : info.dtype=.floating .fp32 → tensor.dtype=.fp32
  device : tensor.device=info.device.index
structure Binding (x w : TensorInfo) (input weight : Whole.Tensor) (live : Nat → Prop) : Prop where
  inputExisted : live input.allocation
  weightExisted : live weight.allocation
  input : Observes x input
  weight : Observes w weight

variable {x w : TensorInfo} {input weight : Whole.Tensor} {live : Nat → Prop}

def Binding.domain (b : Binding x w input weight live) (n : NormalizedInfo) (eps : EpsilonInfo)
    (c : InputContract x w n eps) : Whole.InputDomain input weight n.dimensions live where
  rank := by simpa [b.input.shape] using c.rank
  suffix := by simpa [Whole.dim,dim,b.input.shape] using c.suffix
  weightShape := b.weight.shape.trans c.weightShape
  inputDType := b.input.dtype c.input.dtype
  weightDType := b.weight.dtype c.weight.dtype
  sameDevice := by rw [b.input.device,b.weight.device,c.device]
  inputExisted := b.inputExisted
  weightExisted := b.weightExisted

theorem Binding.rows (b : Binding x w input weight live) (n : List Nat) :
    Whole.rows input n=rows x n := by simp [Whole.rows,Whole.dim,VeriTile.Bench.Repairs.RmsNormTriton.rows,VeriTile.Bench.Repairs.RmsNormTriton.dim,b.input.shape]
theorem block_eq (n : Nat) : block n=Whole.nextPower n := rfl

#axiomsClean checkInput_iff
#axiomsClean checkPointers_iff
end
end VeriTile.Bench.Repairs.RmsNormTriton
