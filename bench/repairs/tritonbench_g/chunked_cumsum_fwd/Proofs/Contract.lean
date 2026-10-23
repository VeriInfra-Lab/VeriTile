import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Contract
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.HostStorage

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd
open VeriTile.Torch VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd RawBoundary HostModel HostStorage
noncomputable section

theorem tensorCheck_iff (device : Device) (x : TensorMetadata) :
    tensorCheck device x=Bool.true ↔ TensorContract device x := by
  have fields : TensorContract device x ↔ x.layout=.strided ∧ x.dtype=.floating .fp32 ∧
      x.device.kind="cuda" ∧ x.device=device ∧ x.isNeg=Bool.false ∧ (0<x.numel → x.dataPtr.toNat%4=0) := by
    constructor
    · intro a; exact ⟨a.layout,a.dtype,a.cuda,a.device,a.noNeg,a.alignment⟩
    · rintro ⟨a,b,c,d,e,f⟩; exact ⟨a,b,c,d,e,f⟩
  rw [fields]
  simp [tensorCheck,TensorMetadata.isCuda,Nat.pos_iff_ne_zero]
  tauto

theorem checkInput_iff (dt rate : TensorMetadata) (bias : Option TensorMetadata)
    (chunk : Int) (integerChunk booleanSoftplus floatLimits : Bool) :
    checkInput dt rate bias chunk integerChunk booleanSoftplus floatLimits=.ok () ↔
      InputContract dt rate bias chunk integerChunk booleanSoftplus floatLimits := by
  have fields : InputContract dt rate bias chunk integerChunk booleanSoftplus floatLimits ↔
      (∀ x∈tensors dt rate bias, TensorContract dt.device x) ∧ dt.shape.length=3 ∧
      rate.shape=[dt.shape.getD 2 0] ∧ (∀ x∈bias.toList, x.shape=[dt.shape.getD 2 0]) ∧
      integerChunk=Bool.true ∧ 0<chunk ∧ chunk<2^63 ∧ chunk≤1048576 ∧ booleanSoftplus=Bool.true ∧ floatLimits=Bool.true ∧
      (∀ n∈dt.shape,n<2^63) ∧ (0<dt.numel → dt.shape.getD 0 0<2^31 ∧
        (dt.shape.getD 1 0+chunk.toNat-1)/chunk.toNat≤65535 ∧ dt.shape.getD 2 0≤65535) := by
    constructor
    · intro a; exact ⟨a.tensors,a.rank,a.rateShape,a.biasShape,a.chunkType,a.chunkPositive,
        a.chunkBound,a.chunkBlock,a.softplusType,a.limitsType,a.shapeBounds,a.grid⟩
    · rintro ⟨a,b,c,d,e,f,g,h,i,j,k,l⟩; exact ⟨a,b,c,d,e,f,g,h,i,j,k,l⟩
  rw [fields]
  simp [checkInput,ite_eq_iff,tensorCheck_iff]
  tauto

def noNegative : NegativeFlags := ⟨Bool.false,Bool.false,Bool.false⟩

/-- Metadata describes exactly the source call. Strides belong to its original
Tensor observations; the separate Backings adapter binds actual data pointers. -/
structure TensorBinding (dt rate : TensorMetadata) (bias : Option TensorMetadata) (call : Call α) : Prop where
  dtShape : call.dt.shape=dt.shape
  rateShape : call.rate.shape=rate.shape
  biasPresent : call.bias.isSome=bias.isSome
  biasShape : ∀ x y,call.bias=some x → bias=some y → x.shape=y.shape
  dtDevice : call.dt.device=dt.device.index
  rateDevice : call.rate.device=rate.device.index
  biasDevice : ∀ x y,call.bias=some x → bias=some y → x.device=y.device.index

/-- Every bound used by the unsigned source masks follows from checked
dimension and chunk metadata, rather than the allocator interface. -/
theorem InputContract.geometrySizes
    (contract : InputContract dt rate bias call.chunk integerChunk booleanSoftplus floatLimits)
    (binding : TensorBinding dt rate bias call) (g : HostModel.Geometry call) :
    Native.Sizes g.host := by
  have shape : dt.shape=[g.host.batches,g.host.sequence,g.host.heads] :=
    binding.dtShape.symm.trans g.shape
  constructor
  · exact contract.shapeBounds _ (by rw [shape]; simp)
  · exact contract.shapeBounds _ (by rw [shape]; simp)
  · rw [g.chunk]
    have positive := contract.chunkPositive
    have bound := contract.chunkBound
    omega

theorem InputContract.geometryArange
    (contract : InputContract dt rate bias call.chunk integerChunk booleanSoftplus floatLimits)
    (g : HostModel.Geometry call) : g.host.BC≤1048576 := by
  have chunk : g.host.chunk≤2^20 := by
    rw [g.chunk]
    have positive := contract.chunkPositive
    have bounded := contract.chunkBlock
    omega
  exact Nat.pow_le_pow_right (by decide) (Nat.clog_le_of_le_pow chunk)

def PointerBinding (dt rate : TensorMetadata) (bias : Option TensorMetadata)
    {call : Call α} {logical : Logical α} {entry : PhysicalMemory α} {old : Nat → Prop}
    {rt : HostModel.Runtime call logical entry old} (storage : OrdinaryStorage rt) : Prop :=
  ∀ g a p ha hp, let b := storage g a p ha hp
    (0<dt.numel → b.dt.dataPtr=dt.dataPtr.toNat) ∧
    (0<rate.numel → b.rate.dataPtr=rate.dataPtr.toNat) ∧
    (∀ x∈bias.toList,0<x.numel → b.bias.dataPtr=x.dataPtr.toNat)

def guarded (dt rate : TensorMetadata) (bias : Option TensorMetadata) (chunk : Int)
    (integerChunk booleanSoftplus floatLimits : Bool) (body : HostModel.Outcome α) : Except InputError (HostModel.Outcome α) := do
  checkInput dt rate bias chunk integerChunk booleanSoftplus floatLimits
  return body

theorem guarded_eq (h : InputContract dt rate bias chunk integerChunk booleanSoftplus floatLimits)
    (body : HostModel.Outcome α) : guarded dt rate bias chunk integerChunk booleanSoftplus floatLimits body=.ok body := by
  simp [guarded,(checkInput_iff dt rate bias chunk integerChunk booleanSoftplus floatLimits).mpr h]

#axiomsClean checkInput_iff
#axiomsClean InputContract.geometrySizes
#axiomsClean InputContract.geometryArange
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd
