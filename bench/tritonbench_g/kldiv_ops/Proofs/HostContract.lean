import bench.tritonbench_g.kldiv_triton.Contract

namespace VeriTile.Bench.Repairs.KldivOps.Contiguous
noncomputable section
abbrev Tensor := KldivTriton.Contiguous.Tensor
abbrev Argument := KldivTriton.Contiguous.Argument
abbrev TensorContract := KldivTriton.Contiguous.TensorContract
abbrev PairContract := KldivTriton.Contiguous.PairContract
abbrev BackwardContract := KldivTriton.Contiguous.BackwardContract

/-- The epsilon classification checks Python float, including all exceptional
float values. Runtime FP32 scalar binding is a separate ordinary ABI action. -/
structure ForwardContract (a b : Tensor) (epsilonIsFloat : Bool) : Prop extends PairContract a b where
  epsilon : epsilonIsFloat = Bool.true

def checkForward (a b : Argument) (flag : Option Bool) (mode : String)
    (epsilonIsFloat : Bool) : Except String Unit := do
  KldivTriton.Contiguous.checkForward a b flag mode
  if epsilonIsFloat then .ok () else .error "epsilon type"

abbrev checkBackward := KldivTriton.Contiguous.checkBackward

def AdmittedForward (a b : Argument) (flag : Option Bool) (mode : String)
    (epsilonIsFloat : Bool) : Prop :=
  match a,b with
  | .tensor a,.tensor b => ForwardContract a b epsilonIsFloat ∧ flag.isSome = Bool.true ∧ KldivTriton.Contiguous.validMode mode
  | _,_ => False

abbrev AdmittedBackward := KldivTriton.Contiguous.AdmittedBackward
end
end VeriTile.Bench.Repairs.KldivOps.Contiguous
