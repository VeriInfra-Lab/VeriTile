import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.PythonBodyParsing
open VeriTile.Triton

def scalarSignature : CallSignature where
  scalars := [("x",.integer),("y",.integer)]

/-- Adjacent Python minus operators cannot become a Lean line comment. -/
def adjacentMinus : ComputeKernel :=
  with_triton_signature scalarSignature in triton {
    z = x--y
  }
def explicitMinus : ComputeKernel := triton {
  tl.param x : tl.int64
  tl.param y : tl.int64
  z = x - (-y)
}
example : adjacentMinus=explicitMinus := by rfl

/-- Python comment braces do not terminate the enclosing source quotation.
Elif lowering preserves the complete nested branch tree. -/
def branches : ComputeKernel :=
  with_triton_signature scalarSignature in triton {
    # } This is a Python comment, including its brace.
    if x < 0:
        z = x
    elif x < 1:
        z = y
    else:
        z = x + y
  }
def explicitBranches : ComputeKernel := triton {
  tl.param x : tl.int64
  tl.param y : tl.int64
  if x < $( (0 : Int) ) {
    z = x
  } else {
    if x < $( (1 : Int) ) {
      z = y
    } else {
      z = x + y
    }
  }
}
example : branches=explicitBranches := by rfl

/-- Layout handling retains statements after an entire while suite. -/
def pythonWhile (stride : Nat) : ComputeKernel :=
  with_triton_signature ({scalars:=[("N",.integer)],constexpr:=["stride"]} : CallSignature) in triton {
    N = tl.cast(N, tl.uint64)
    i = tl.full((), 0, tl.uint64)
    while i < N:
        i += stride
    after = i
  }
def explicitWhile (stride : Nat) : ComputeKernel :=
  with_triton_signature ({scalars:=[("N",.integer)],constexpr:=["stride"]} : CallSignature) in triton {
    N = tl.cast(N, tl.uint64)
    i = tl.full((), 0, tl.uint64)
    while i < N {
      i += stride
    }
    after = i
  }
example (stride : Nat) : pythonWhile stride=explicitWhile stride := by rfl

def integerLiterals : ComputeKernel :=
  with_triton_signature scalarSignature in triton {
    left = 0 == x
    negative = x >= -1
  }
def explicitIntegerLiterals : ComputeKernel := triton {
  tl.param x : tl.int64
  tl.param y : tl.int64
  left = $( (0 : Int) ) == x
  negative = x >= $( (-1 : Int) )
}
example : integerLiterals=explicitIntegerLiterals := by rfl

#guard (DSL.RawSourceText.normalize "{s = 'False#}'\nz = True\nx = 1.}") ==
  .ok "s = \"False#}\"\nz = true\nx = 1.0"

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Tests.PythonBodyParsing
