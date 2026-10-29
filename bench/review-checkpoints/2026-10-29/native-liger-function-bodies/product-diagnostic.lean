import bench.tests.IntegerLiteralAssignmentBody
namespace VeriTile.Tests.TypedIntegerOperandBody
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  typedFP32Storage := Bool.true
/-- Event sentinels distinguish operand order, integer channel and precision. -/
def model : Algebra Int :=
  { IntegerLiteralAssignmentBody.model with
    binary := fun p _ _ a b => if p = some .fp32 then 100*a+b else 5000 }
def initial : State Int :=
  { IntegerLiteralAssignmentBody.initial with
    mem := fun _ _ => .mk .fp32 10
    pids := fun _ => 3 }

def rightInteger (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(X)
    n = tl.full((), 2, tl.int64)
    result = x / n
    tl.store(Y, result)
  }
def leftInteger (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(X)
    n = tl.full((), 2, tl.int64)
    result = n / x
    tl.store(Y, result)
  }
def naturalInteger (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(X)
    n = tl.program_id(0)
    result = x / n
    tl.store(Y, result)
  }

def reducedProduct (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    offsets = tl.arange(0, 1)
    x = tl.load(X + offsets)
    wdy = x * x
    n = tl.full((), 2, tl.int64)
    result = tl.sum(wdy, axis=0) / n
    tl.store(Y, result)
  }

/-- The reduction sentinel distinguishes an actual typed FP32 reduction from
an erased precision annotation; the result still divides by a signed tensor. -/
def reductionModel : Algebra Int :=
  { model with reduceSum := fun p {_} _ _ _ _ => if p = some .fp32 then 11 else 9000 }
set_option pp.width 130
#print reducedProduct
end
end VeriTile.Tests.TypedIntegerOperandBody
