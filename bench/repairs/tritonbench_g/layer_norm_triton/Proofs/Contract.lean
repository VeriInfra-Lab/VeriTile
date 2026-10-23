import bench.repairs.tritonbench_g.layer_norm_triton.Contract
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic
namespace VeriTile.Bench.Repairs.LayerNormTriton.Contract
open VeriTile.Torch

theorem tensorCheck_iff (m : TensorMetadata) : tensorCheck m=Bool.true ↔ TensorContract m := by
  simp [tensorCheck,TensorContract,and_assoc]
  omega

theorem epsilonCheck_iff (eps : Option Float) : epsilonCheck eps=Bool.true ↔ EpsilonContract eps := by
  cases eps <;> simp [epsilonCheck,EpsilonContract,and_assoc]

theorem shapeCheck_iff (x w b : TensorMetadata) (ns : Option (List Nat)) :
    shapeCheck x w b ns=Bool.true ↔ ShapeContract x w b ns := by
  cases ns with
  | none => simp [shapeCheck,ShapeContract]
  | some ns =>
    have unpack : ShapeContract x w b (some ns) ↔ ns≠[] ∧ ns.length≤x.shape.length ∧
        x.shape.drop (x.shape.length-ns.length)=ns ∧ w.shape=ns ∧ b.shape=ns := by
      constructor
      · rintro ⟨other,he,h⟩
        cases he
        exact h
      · intro h
        exact ⟨ns,rfl,h⟩
    rw [unpack]
    cases ns <;> simp [shapeCheck,and_assoc]

theorem inputCheck_iff (x w b : TensorMetadata) (ns : Option (List Nat)) (eps : Option Float) :
    inputCheck x w b ns eps=Bool.true ↔ InputContract x w b ns eps := by
  simp [inputCheck,InputContract,tensorCheck_iff,epsilonCheck_iff,shapeCheck_iff,and_assoc]

theorem gradientCheck_iff (dy x : TensorMetadata) (shape : List Nat) :
    gradientCheck dy x shape=Bool.true ↔ GradientContract dy x shape := by
  simp [gradientCheck,GradientContract,tensorCheck_iff,and_assoc]

theorem checkArguments_iff (x w b : ArgumentMetadata) (ns : Option (List Nat)) (eps : Option Float) :
    checkArguments x w b ns eps=Bool.true ↔ AdmittedArguments x w b ns eps := by
  cases x <;> cases w <;> cases b <;> simp [checkArguments,AdmittedArguments,inputCheck_iff]

#axiomsClean inputCheck_iff
#axiomsClean gradientCheck_iff
#axiomsClean checkArguments_iff
end VeriTile.Bench.Repairs.LayerNormTriton.Contract
