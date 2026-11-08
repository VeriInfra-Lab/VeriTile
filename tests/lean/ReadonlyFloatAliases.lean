import VeriTile.Triton.DSL.ReadonlyFloatAliases
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.ReadonlyFloatAliases
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1500000
set_option linter.unusedVariables false

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := []
  constexpr := ["B"]
def source (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    c = -1.75
    d = c
    lane = tl.arange(0, B)
    x = tl.load(X + lane)
    y = -d * x + c * x
    tl.store(Y + lane, y)
  }
def expected (X Y : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region X=tl.float32, Y=tl.float32
  c = -1.75
  d = (-1.75)
  lane = tl.arange(0, $(B))
  x = tl.load(X + lane)
  y = -(-1.75) * x + (-1.75) * x
  tl.store(Y + lane, y)
}
/-- Both original bindings remain; literal uses keep the consuming tensor's
FP32 context and do not become untagged real arithmetic. -/
theorem complete_program (X Y : RegionName) (B : Nat) :
    source X Y B=expected X Y B := rfl

def halfSignature : CallSignature where
  pointers := [("X",.fp16),("Y",.fp16)]
  scalars := []
  constexpr := ["B"]
def halfSource (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call halfSignature call_args [] in triton {
    c = 1.75
    lane = tl.arange(0, B)
    x = tl.load(X + lane)
    y = c * x
    tl.store(Y + lane, y)
  }
def halfExpected (X Y : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region X=tl.float16, Y=tl.float16
  c = 1.75
  lane = tl.arange(0, $(B))
  x = tl.load(X + lane)
  y = (1.75) * x
  tl.store(Y + lane, y)
}
/-- The alias is a weak scalar; it must not widen a half tensor to FP32. -/
theorem half_context (X Y : RegionName) (B : Nat) :
    halfSource X Y B=halfExpected X Y B := rfl

def signedZeroSource (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    c = -0.0
    lane = tl.arange(0, B)
    x = tl.load(X + lane)
    y = c * x
    tl.store(Y + lane, y)
  }
def signedZeroExpected (X Y : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region X=tl.float32, Y=tl.float32
  c = -0.0
  lane = tl.arange(0, $(B))
  x = tl.load(X + lane)
  y = (-0.0) * x
  tl.store(Y + lane, y)
}
theorem signed_zero_literal (X Y : RegionName) (B : Nat) :
    signedZeroSource X Y B=signedZeroExpected X Y B := rfl

def castSource (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    c = 1.75
    lane = tl.arange(0, B)
    x = tl.load(X + lane)
    y = tl.cast(c, tl.float64) * x
    tl.store(Y + lane, y)
  }
def castExpected (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    c = 1.75
    lane = tl.arange(0, B)
    x = tl.load(X + lane)
    y = tl.cast((1.75), tl.float64) * x
    tl.store(Y + lane, y)
  }
theorem explicit_cast_retained (X Y : RegionName) (B : Nat) :
    castSource X Y B=castExpected X Y B := rfl

def keywordSource (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    other = 1.75
    lane = tl.arange(0, B)
    x = tl.load(X + lane, mask=lane < B, other=other)
    tl.store(Y + lane, x)
  }
def keywordExpected (X Y : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region X=tl.float32, Y=tl.float32
  other = 1.75
  lane = tl.arange(0, $(B))
  x = tl.load(X + lane, mask=lane < $(B), other=(1.75))
  tl.store(Y + lane, x)
}
/-- A keyword name is syntax, while its value may use a literal alias. -/
theorem keyword_hygiene (X Y : RegionName) (B : Nat) :
    keywordSource X Y B=keywordExpected X Y B := rfl

def mutableSource (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    c = 1.75
    c = 2.5
    lane = tl.arange(0, B)
    x = tl.load(X + lane)
    y = c * x
    tl.store(Y + lane, y)
  }
def mutableExpected (X Y : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region X=tl.float32, Y=tl.float32
  c = 1.75
  c = 2.5
  lane = tl.arange(0, $(B))
  x = tl.load(X + lane)
  y = c * x
  tl.store(Y + lane, y)
}
/-- Mutable locals retain the ordinary elaborator's actual register uses. -/
theorem mutable_not_inlined (X Y : RegionName) (B : Nat) :
    mutableSource X Y B=mutableExpected X Y B := rfl

example (X Y : RegionName) (B : Nat) : True := by
  fail_if_success
    let bad : ComputeKernel := with_triton_call signature call_args [] in triton {
      lane = tl.arange(0, B)
      x = tl.load(X + lane)
      y = c * x
      c = 1.75
      tl.store(Y + lane, y)
    }
  trivial

#axiomsClean complete_program
#axiomsClean half_context
#axiomsClean signed_zero_literal
#axiomsClean explicit_cast_retained
#axiomsClean keyword_hygiene
#axiomsClean mutable_not_inlined
end
end VeriTile.Tests.ReadonlyFloatAliases
