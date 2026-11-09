import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural

namespace VeriTile.Tests.PreciseDivisionAndInfinity
open VeriTile.Triton FP.Structural
noncomputable section

def signature : CallSignature where
  pointers := [("x",.fp32),("y",.fp32),("out",.fp32)]
  scalars := []
  constexpr := []

def precise (x y out : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    left = tl.load(x)
    right = tl.load(y)
    result = tl.div_rn(left, right)
    tl.store(out, result)
  }

def ordinary (x y out : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    left = tl.load(x)
    right = tl.load(y)
    result = left / right
    tl.store(out, result)
  }

def preciseOp {dtype : TileDType} {shape : TileShape} : Op dtype shape → Bool
  | .divRN .. => Bool.true
  | _ => Bool.false

def preciseExpr {dtype : TileDType} {shape : TileShape} : ComputeExpr dtype shape → Bool
  | .compute operation => match operation with
    | .alg precision operation => precision == .fp32 && preciseOp operation
    | _ => Bool.false
  | _ => Bool.false

def preciseStatement : ComputeStmt → Bool
  | .assign _ _ _ expression => preciseExpr expression
  | _ => Bool.false

example (x y out : RegionName) :
    preciseStatement ((precise x y out).surfaceBody.getD 2 (.effectMarker "missing")) = Bool.true := by rfl
example (x y out : RegionName) :
    preciseStatement ((ordinary x y out).surfaceBody.getD 2 (.effectMarker "missing")) = Bool.false := by rfl

/-- Availability is required even for an empty tile. -/
example {α : Type} [Inhabited α] (rest : Algebra α) (s : State α) :
    FP.Structural.evalOp {rest with divRN := none} (some .fp32)
      (.divRN (IntegerExpr.sameShape [0]) (.full [0] (.const 1)) (.full [0] (.const 2))) s = none := by
  simp [evalOp_unfold]

/-- The dedicated callback supplies the result independently of ordinary division. -/
example {α : Type} [Inhabited α] (rest : Algebra α) (f : α → α → α) (s : State α) :
    FP.Structural.evalOp {rest with divRN := some f} (some .fp32)
      (.divRN .nil (.const 1) (.const 2)) s =
      some (fun _ => f (rest.literal (some .fp32) .real 1) (rest.literal (some .fp32) .real 2)) := by
  simp [evalOp_unfold]
  rfl

def outputSignature : CallSignature where
  pointers := [("out",.fp32)]
  scalars := []
  constexpr := []

def positiveInfinity (out : RegionName) : ComputeKernel :=
  with_triton_signature outputSignature in triton {
    value = float('inf')
    tl.store(out, value)
  }

def encodedInfinity (out : RegionName) : ComputeKernel :=
  with_triton_signature outputSignature in triton {
    value = tl.bitcast(tl.full([], 2139095040, dtype=tl.uint32), tl.float32)
    tl.store(out, value)
  }

/-- The positive literal uses its exact binary32 word, with no negation callback. -/
example (out : RegionName) : positiveInfinity out = encodedInfinity out := by rfl

/-- error: float(...): only literal inf and -inf are supported -/
#guard_msgs in
example (out : RegionName) : ComputeKernel :=
  with_triton_signature outputSignature in triton {
    value = float('not-inf')
    tl.store(out, value)
  }

def wideSignature : CallSignature where
  pointers := [("x",.fp64),("y",.fp64),("out",.fp64)]
  scalars := []
  constexpr := []

/-- error: tl.div_rn requires FP32 operands -/
#guard_msgs in
example (x y out : RegionName) : ComputeKernel :=
  with_triton_signature wideSignature in triton {
    left = tl.load(x)
    right = tl.load(y)
    result = tl.div_rn(left, right)
    tl.store(out, result)
  }

end
end VeriTile.Tests.PreciseDivisionAndInfinity
