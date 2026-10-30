import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.SourceConstantBody
open VeriTile.Triton
noncomputable section

def signature : CallSignature where
  pointers := [("X", .fp32), ("Y", .fp32)]
  constexpr := ["B"]
  sourceConstants := [("C", "1.44269504")]

def literalSignature : CallSignature where
  pointers := [("X", .fp32), ("Y", .fp32)]
  constexpr := ["B"]

def source (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    acc = tl.zeros([B], dtype=tl.float32)
    for i in range(2):
        x = tl.load(X + tl.arange(0, B))
        acc += tl.math.exp2(x) * C
    tl.store(Y + tl.arange(0, B), acc)
  }

def literal (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature literalSignature in triton {
    acc = tl.zeros([B], dtype=tl.float32)
    for i in range(2):
        x = tl.load(X + tl.arange(0, B))
        acc += tl.math.exp2(x) * 1.44269504
    tl.store(Y + tl.arange(0, B), acc)
  }

/-- The module literal is weak at the original operand position. It adds no
register assignment, cast or arithmetic, including inside the loop. -/
theorem same_complete_source (X Y : RegionName) (B : Nat) :
    source X Y B = literal X Y B := rfl

/-- The FP32 exp2 result keeps the accumulator dtype stable without an
invented source cast or redundant implicit store conversion. -/
theorem statement_count (X Y : RegionName) (B : Nat) :
    (source X Y B).surfaceBody.length = 3 := rfl

def computed : CallSignature where
  pointers := [("Y", .fp32)]
  sourceConstants := [("C", "1.0 + 0.44269504")]

/-- error: source constant requires a numeric literal -/
#guard_msgs in
def rejectedComputed (Y : RegionName) : ComputeKernel :=
  with_triton_signature computed in triton {
    tl.store(Y, C)
  }

/-- error: source constants cannot be rebound -/
#guard_msgs in
def rejectedMutation (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    C = 1.0
    tl.store(Y, C)
  }

/-- error: source constants cannot be rebound -/
#guard_msgs in
def rejectedIterator (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    for C in range(2):
        tl.store(Y, C)
  }

#axiomsClean same_complete_source
#axiomsClean statement_count
end
end VeriTile.Tests.SourceConstantBody
