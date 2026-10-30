import bench.repairs.tritonbench_g.rmsnorm_triton.Contract
import bench.repairs.tritonbench_g.rmsnorm_triton.Kernels
import bench.tritonbench_g.rmsnorm_triton.Proofs.WholePublic

namespace VeriTile.Bench.Repairs.RmsnormTriton
open VeriTile.Torch VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
noncomputable section
set_option maxHeartbeats 3000000

theorem checkLaunchInput_iff (x w : TensorInfo) :
    checkLaunchInput x w=.ok () ↔ LaunchInput x w := by
  have fields : LaunchInput x w ↔ batches x<2^31 ∧ rows x≤65535 ∧ padded x<2^31 ∧
      (x.stride 0<2^31 ∧ x.stride 1<2^31 ∧ x.stride 2<2^31 ∧ w.stride 0<2^31) ∧
      (maxOffset x x<2^31 ∧ maxWeight x w<2^31) ∧
      (x.dataPtr.toNat%4=0 ∧ w.dataPtr.toNat%4=0) ∧
      (x.dataPtr.toNat+4*maxOffset x x<2^64 ∧ w.dataPtr.toNat+4*maxWeight x w<2^64) := by
    constructor
    · intro h; exact ⟨h.gridX,h.gridY,h.loop,h.strides,h.offsets,h.alignment,h.pointers⟩
    · rintro ⟨a,b,c,d,e,f,g⟩; exact ⟨a,b,c,d,e,f,g⟩
  rw [fields]
  simp [checkLaunchInput,ite_eq_iff]
  tauto

theorem checkTensorInput_iff (x w : TensorInfo) (eps : EpsilonInfo) :
    checkTensorInput x w eps=.ok () ↔ InputContract x w eps := by
  have fields : InputContract x w eps ↔
      (x.layout=.strided ∧ x.dtype=.floating .fp32 ∧ x.device.kind="cuda") ∧
      (w.layout=.strided ∧ w.dtype=.floating .fp32 ∧ w.device.kind="cuda") ∧
      x.device=w.device ∧ x.shape.length=3 ∧ 0<width x ∧ w.shape=[width x] ∧
      (x.isContiguous=Bool.true ∧ x.isNeg=Bool.false ∧ x.isConj=Bool.false) ∧
      (w.isContiguous=Bool.true ∧ w.isNeg=Bool.false ∧ w.isConj=Bool.false) ∧
      eps.pythonFloat=Bool.true ∧ eps.finite=Bool.true ∧ 0<eps.value ∧
      (x.numel≠0 → LaunchInput x w) := by
    constructor
    · intro h
      exact ⟨⟨h.input.layout,h.input.dtype,h.input.cuda⟩,⟨h.weight.layout,h.weight.dtype,h.weight.cuda⟩,
        h.device,h.rank,h.positive,h.weightShape,
        ⟨h.inputDense.contiguous,h.inputDense.negative,h.inputDense.conjugate⟩,
        ⟨h.weightDense.contiguous,h.weightDense.negative,h.weightDense.conjugate⟩,
        h.epsilonType,h.epsilonFinite,h.epsilonPositive,h.launch⟩
    · rintro ⟨⟨a,b,c⟩,⟨d,e,f⟩,g,h,i,j,⟨k,l,m⟩,⟨n,o,p⟩,q,r,s,t⟩
      exact ⟨⟨a,b,c⟩,⟨d,e,f⟩,g,h,i,j,⟨k,l,m⟩,⟨n,o,p⟩,q,r,s,t⟩
  rw [fields]
  simp [checkTensorInput,TensorMetadata.isCuda,ite_eq_iff,checkLaunchInput_iff,Nat.pos_iff_ne_zero]
  tauto

theorem checkOutput_iff (x out : TensorInfo) :
    checkOutput x out=.ok () ↔ LaunchOutput x out := by
  have fields : LaunchOutput x out ↔
      (out.stride 0<2^31 ∧ out.stride 1<2^31 ∧ out.stride 2<2^31) ∧
      maxOffset x out<2^31 ∧ out.dataPtr.toNat%4=0 ∧ out.dataPtr.toNat+4*maxOffset x out<2^64 := by
    constructor
    · intro h; exact ⟨h.strides,h.offset,h.alignment,h.pointer⟩
    · rintro ⟨a,b,c,d⟩; exact ⟨a,b,c,d⟩
  rw [fields]
  simp [checkOutput,ite_eq_iff]
  tauto

theorem checkInput_iff (x w : Argument) (eps : EpsilonInfo) :
    checkInput x w eps=.ok () ↔ Admitted x w eps := by
  cases x <;> cases w <;> simp [checkInput,Admitted,checkTensorInput_iff]

theorem shape_three (x : TensorInfo) (rank : x.shape.length=3) :
    x.shape=[batches x,rows x,width x] := by
  rcases x with ⟨⟨dtype,layout,device,shape,contiguous,negative,ptr⟩,stride,offset,conj⟩
  cases shape with
  | nil => simp at rank
  | cons b bs => cases bs with
    | nil => simp at rank
    | cons m ms => cases ms with
      | nil => simp at rank
      | cons k ks => cases ks with
        | nil => rfl
        | cons v vs => simp at rank

structure Observes (info : TensorInfo) (t : Tensor) (byteBase : Nat → Nat) : Prop where
  shape : t.shape=info.shape
  stride : t.stride=info.stride
  base : t.base=info.storageOffset
  dtype : info.dtype=.floating .fp32 → t.dtype=.fp32
  device : t.device=info.device.index
  pointer : info.numel≠0 → byteBase t.allocation+4*t.base=info.dataPtr.toNat

/-- Ordinary live Tensor and allocator facts; no execution, padded-address,
index-ABI, or numerical-result premises. -/
structure MemoryAPI (p : Parameters) (live : Nat → Prop) where
  capacity : Nat → Nat
  byteBase : Nat → Nat
  inputLive : live p.input.allocation
  weightLive : live p.weights.allocation
  outputFresh : ¬live p.output.allocation
  outputUnique : OutputUnique p
  inputBounds : ∀ b m k, b<p.batches → m<p.rows → k<p.width →
    inputOffset p b m k<capacity p.input.allocation
  weightBounds : ∀ k,k<p.width → weightOffset p k<capacity p.weights.allocation
  outputBounds : ∀ b m k,b<p.batches → m<p.rows → k<p.width →
    outputOffset p b m k<capacity p.output.allocation
  allocationEnd : ∀ t∈[p.input,p.weights,p.output],byteBase t.allocation+4*capacity t.allocation≤2^64
  freshBytes : ∀ a,live a →
    byteBase p.output.allocation+4*capacity p.output.allocation≤byteBase a ∨
    byteBase a+4*capacity a≤byteBase p.output.allocation

/-- Fresh output allocation supplies separation; input and weight may alias. -/
def MemoryAPI.separation (api : MemoryAPI p live) : Separation p where
  input := fun same => api.outputFresh (same ▸ api.inputLive)
  weights := fun same => api.outputFresh (same ▸ api.weightLive)

structure Binding (x w out : TensorInfo) (eps : EpsilonInfo) (p : Parameters)
    (byteBase : Nat → Nat) : Prop where
  batch : p.batches=batches x
  row : p.rows=rows x
  column : p.width=width x
  epsilon : p.eps=eps.value
  input : Observes x p.input byteBase
  weight : Observes w p.weights byteBase
  output : Observes out p.output byteBase
  outputShape : out.shape=x.shape

def Binding.domain (x w out : TensorInfo) (eps : EpsilonInfo) (c : InputContract x w eps)
    (p : Parameters) (api : MemoryAPI p live) (b : Binding x w out eps p api.byteBase) : Domain p where
  inputShape := by rw [b.input.shape,shape_three x c.rank,b.batch,b.row,b.column]
  weightShape := by rw [b.weight.shape,c.weightShape,b.column]
  inputFP32 := b.input.dtype c.input.dtype
  weightFP32 := b.weight.dtype c.weight.dtype
  sameDevice := by rw [b.weight.device,b.input.device,c.device]
  epsilonPositive := by rw [b.epsilon]; exact c.epsilonPositive
  capacity := api.capacity
  inputBounds := api.inputBounds
  weightBounds := api.weightBounds
  outputBounds := api.outputBounds

theorem Binding.numel (x w out : TensorInfo) (eps : EpsilonInfo) (c : InputContract x w eps)
    (p : Parameters) (byteBase : Nat → Nat) (b : Binding x w out eps p byteBase) :
    x.numel=p.batches*p.rows*p.width := by
  simp [TensorMetadata.numel,shape_three x c.rank,b.batch,b.row,b.column,Nat.mul_assoc]

#axiomsClean checkTensorInput_iff
#axiomsClean checkOutput_iff
#axiomsClean checkInput_iff
end
end VeriTile.Bench.Repairs.RmsnormTriton
