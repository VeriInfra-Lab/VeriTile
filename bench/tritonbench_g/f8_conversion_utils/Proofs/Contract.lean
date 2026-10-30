import bench.tritonbench_g.f8_conversion_utils.Contract
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.F8ConversionUtils.Checked
open VeriTile.Torch

theorem input_contract_iff (d : Direction) (format : FormatArgument) (x : TensorMetadata) :
    InputContract d format x ↔ x.layout = .strided ∧ inputDType d x ∧ format ≠ .other ∧
      x.device.kind = "cuda" ∧ x.isContiguous = Bool.true ∧ x.isNeg = Bool.false ∧
      (0 < x.numel → x.dataPtr.toNat % inputWidth d x = 0) := by
  constructor
  · intro h; exact ⟨h.layout,h.dtype,h.format,h.cuda,h.contiguous,h.noNeg,h.alignment⟩
  · rintro ⟨hl,ht,hf,hd,hc,hn,ha⟩; exact ⟨hl,ht,hf,hd,hc,hn,ha⟩

theorem checkTensor_iff (d : Direction) (format : FormatArgument) (x : TensorMetadata) :
    checkTensor d format x = .ok () ↔ InputContract d format x := by
  simp [checkTensor,input_contract_iff,TensorMetadata.isCuda,ite_eq_iff]

theorem checkInputs_iff (d : Direction) (format : FormatArgument) (x : ArgumentMetadata) :
    checkInputs d format x = .ok () ↔ AdmittedArguments d format x := by
  cases x with
  | nonTensor => simp [checkInputs,AdmittedArguments]
  | tensor x => exact checkTensor_iff d format x

theorem profile_width (d : Direction) (format : FormatArgument) (x : TensorMetadata) :
    (profile d format x).inputWidth = inputWidth d x := by
  cases d <;> cases format <;> simp [profile,inputWidth,Source.Profile.inputWidth]
  all_goals split_ifs <;> rfl

theorem profile_twice (d : Direction) (format : FormatArgument) (x : TensorMetadata) :
    (profile d format x).twice = d.doubleStore := by
  cases d <;> cases format <;> simp [profile,Source.Profile.twice,Direction.doubleStore]
  all_goals split_ifs <;> rfl

#axiomsClean checkInputs_iff
#axiomsClean profile_width
end VeriTile.Bench.Repairs.F8ConversionUtils.Checked
