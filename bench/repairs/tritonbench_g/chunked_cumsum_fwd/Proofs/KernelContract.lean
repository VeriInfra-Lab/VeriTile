import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Contract
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic.Tauto

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.Contract
open VeriTile.Torch

/-! Pure equivalence of the existing metadata checker and input contract.
No runtime, allocation or execution outcome is introduced. -/
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

#axiomsClean tensorCheck_iff
#axiomsClean checkInput_iff
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.Contract
