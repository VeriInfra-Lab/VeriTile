import bench.tritonbench_g.fused_recurrent_hgrn.Kernels
import bench.tritonbench_g.fused_recurrent_hgrn.Sources.Mathematical_FPStages

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn
open VeriTile.Triton FP.Structural
noncomputable section

def forwardPrologue (T D BD : Nat) (U : Bool) := (forward T D BD U Bool.false).surfaceBody.take 9
def forwardBody (D BD : Nat) : List ComputeStmt :=
  match (forward 1 D BD Bool.false Bool.false).surfaceBody.getD 9 (.effectMarker "absent") with
  | ComputeStmt.forRange _ _ _ _ body => body
  | _ => []
def forwardEpilogue (T D BD : Nat) (S : Bool) := (forward T D BD Bool.false S).surfaceBody.drop 10

def backwardPrologue (T D BD : Nat) (V : Bool) := (backward T D BD Bool.false V).surfaceBody.take 6
def backwardBody (T D BD : Nat) (U : Bool) : List ComputeStmt :=
  match (backward T D BD U Bool.false).surfaceBody.getD 6 (.effectMarker "absent") with
  | ComputeStmt.forRange _ _ _ _ body => body
  | _ => []

theorem forward_split (T D BD : Nat) (U S : Bool) :
    (forward T D BD U S).surfaceBody = forwardPrologue T D BD U ++
      [.forRange "_i" 0 T 1 (forwardBody D BD)] ++ forwardEpilogue T D BD S := by rfl

theorem backward_split (T D BD : Nat) (U V : Bool) :
    (backward T D BD U V).surfaceBody = backwardPrologue T D BD V ++
      [.forRange "step" 0 T 1 (backwardBody T D BD U)] := by rfl

/-- Generic source arithmetic. Native lowering uses retained expression graphs, so this algebra does not assert that a fused pair has two separate roundings. -/
structure Ops (α : Type) where
  zero : α
  add : α → α → α
  mul : α → α → α

def algebra {α : Type} (p : Ops α) : Algebra α where
  literal := fun _ _ _ => p.zero
  negInf := p.zero
  binary := fun _ _ op a b => match op with
    | .add => p.add a b | .mul => p.mul a b | _ => p.zero
  unary := fun _ _ _ => p.zero
  cast := fun _ _ _ a => a
  fromNat := fun _ _ => p.zero
  fromInt := fun _ _ => p.zero
  fp32Bits := fun _ => p.zero
  fp32Load := id
  reduceMax := fun _ _ _ _ _ _ => p.zero
  reduceSum := fun _ _ _ _ _ _ => p.zero

end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn
