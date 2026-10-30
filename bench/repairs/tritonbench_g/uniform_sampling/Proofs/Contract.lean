import bench.repairs.tritonbench_g.uniform_sampling.Contract
import VeriTile.Meta.Specification
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic.Tauto

namespace VeriTile.Bench.Repairs.UniformSampling
open VeriTile.Torch VeriTile.Triton
noncomputable section

theorem checkInput_iff (output : TensorMetadata) (generator : GeneratorArgument)
    (lowerIsFloat upperIsFloat : Bool) :
    checkInput output generator lowerIsFloat upperIsFloat=.ok () ↔
      InputContract output generator lowerIsFloat upperIsFloat := by
  have fields : InputContract output generator lowerIsFloat upperIsFloat ↔
      output.layout=.strided ∧ output.dtype=.floating .fp32 ∧ output.device.kind="cuda" ∧
      output.isContiguous=Bool.true ∧ output.isNeg=Bool.false ∧
      (0<output.numel → output.dataPtr.toNat%4=0) ∧ output.numel<2^63 ∧
      generatorCheck output.device generator=Bool.true ∧ lowerIsFloat=Bool.true ∧ upperIsFloat=Bool.true := by
    constructor
    · intro a; exact ⟨a.layout,a.dtype,a.cuda,a.contiguous,a.noNeg,a.alignment,a.extent,a.generator,a.lowerType,a.upperType⟩
    · rintro ⟨a,b,c,d,e,f,g,h,i,j⟩;exact ⟨a,b,c,d,e,f,g,h,i,j⟩
  rw [fields]
  simp [checkInput,ite_eq_iff,TensorMetadata.isCuda,Nat.pos_iff_ne_zero]
  tauto


#axiomsClean checkInput_iff
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.UniformSampling
