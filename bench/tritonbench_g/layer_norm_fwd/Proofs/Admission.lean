import bench.tritonbench_g.layer_norm_fwd.Proofs.Contract

namespace VeriTile.Bench.Repairs.LayerNormFwd.Admission
open Wrapper

 theorem inputCheck_iff (input : Input) (shape : List Nat) (strides : List Int) (device : Nat) :
    inputCheck input shape strides device = Bool.true ↔ InputContract input shape strides device := by
  simp [inputCheck, InputContract, and_assoc, Nat.pos_iff_ne_zero, imp_iff_not_or]

 theorem optionalCheck_iff (input : Option Input) (shape : List Nat) (strides : List Int) (device : Nat) :
    optionalCheck input shape strides device = Bool.true ↔ OptionalContract input shape strides device := by
  cases input <;> simp [optionalCheck, OptionalContract, inputCheck_iff]

 theorem epsilonCheck_iff (value : Option Float) : epsilonCheck value = Bool.true ↔ EpsilonContract value := by
  cases value <;> simp [epsilonCheck, EpsilonContract, and_assoc]

 theorem probabilityCheck_iff (value : Option Float) : probabilityCheck value = Bool.true ↔ ProbabilityContract value := by
  cases value <;> simp [probabilityCheck, ProbabilityContract, and_assoc]

 theorem checkInputs_iff (c : Call) : checkInputs c = Bool.true ↔ InputDomain c := by
  cases hr : c.residual <;>
    simp [checkInputs, InputDomain, inputCheck_iff, optionalCheck_iff,
      epsilonCheck_iff, probabilityCheck_iff, and_assoc, hr]

 theorem optional_dtype (input : Option Input) (shape : List Nat) (strides : List Int) (device : Nat)
    (h : OptionalContract input shape strides device) :
    ∀ value, input.map Input.value = some value → value.dtype = DType.fp32 := by
  intro value hv
  cases hi : input with
  | none => simp [hi] at hv
  | some arg =>
    have he : arg.value = value := by simpa [hi] using hv
    rw [←he]
    exact (h arg hi).2.2.2.1

#axiomsClean inputCheck_iff
#axiomsClean optionalCheck_iff
#axiomsClean epsilonCheck_iff
#axiomsClean probabilityCheck_iff
#axiomsClean checkInputs_iff
end VeriTile.Bench.Repairs.LayerNormFwd.Admission
