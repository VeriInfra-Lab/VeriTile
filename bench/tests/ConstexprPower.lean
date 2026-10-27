import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit

/-! Full-body checks for exact static Python float powers. The resulting
constant remains weak; the surrounding FP32 tensor selects its arithmetic. -/
namespace VeriTile.Tests.ConstexprPower
open VeriTile.Triton
noncomputable section

def profile : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  constexpr := ["B"]
  readPorts := ["X"]
  writePorts := ["Y"]

def source (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature profile in triton {
    scale: tl.constexpr = 2.0 ** (127 - 15)
    offsets = tl.arange(0, B)
    x = tl.load(X + offsets)
    y = x * scale
    tl.store(Y + offsets, y)
  }

def expanded (X Y : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float32
  scale = $(((2 : ℝ)^112 : ℝ))
  offsets = tl.arange(0, $(B))
  x = tl.load(X + offsets)
  y = x * scale
  tl.store(Y + offsets, y)
}

theorem complete_source (X Y : RegionName) (B : Nat) :
    source X Y B = expanded X Y B := rfl

def reciprocalSource (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature profile in triton {
    scale: tl.constexpr = 0.5 ** (1 + 1)
    offsets = tl.arange(0, B)
    x = tl.load(X + offsets)
    tl.store(Y + offsets, x * scale)
  }

def reciprocalExpanded (X Y : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float32
  scale = $((1 / ((2 : ℝ)^2) : ℝ))
  offsets = tl.arange(0, $(B))
  x = tl.load(X + offsets)
  tl.store(Y + offsets, x * scale)
}

theorem reciprocal_source (X Y : RegionName) (B : Nat) :
    reciprocalSource X Y B = reciprocalExpanded X Y B := rfl

/-- error: constexpr float power requires an exactly representable positive binary power literal -/
#guard_msgs in
def rounded (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature profile in triton {
    scale: tl.constexpr = 1.1 ** 2
    offsets = tl.arange(0, B)
    tl.store(Y + offsets, tl.load(X + offsets) * scale)
  }

/-- error: constexpr power is outside the finite nonzero binary64 range -/
#guard_msgs in
def overflow (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature profile in triton {
    scale: tl.constexpr = 2.0 ** 1024
    offsets = tl.arange(0, B)
    tl.store(Y + offsets, tl.load(X + offsets) * scale)
  }

/-- error: per-call specialization does not yet support reassigned scalar arguments -/
#guard_msgs in
def mutated (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature profile in triton {
    scale: tl.constexpr = 2.0 ** 5
    scale = 3.0
    offsets = tl.arange(0, B)
    x = tl.load(X + offsets)
    tl.store(Y + offsets, x * scale)
  }

def exponentProfile : CallSignature :=
  { profile with scalars := [("exponent",.integer)] }

/-- error: constexpr power exponent must be a literal integer expression -/
#guard_msgs in
def runtimeExponent (X Y : RegionName) (B exponent : Nat) : ComputeKernel :=
  with_triton_call exponentProfile call_args [exponent := exponent] in triton {
    scale: tl.constexpr = 2.0 ** exponent
    offsets = tl.arange(0, B)
    x = tl.load(X + offsets)
    tl.store(Y + offsets, x * scale)
  }

/-- error: constexpr power is outside the finite nonzero binary64 range -/
#guard_msgs in
def underflow (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature profile in triton {
    scale: tl.constexpr = 2.0 ** (-1075)
    offsets = tl.arange(0, B)
    x = tl.load(X + offsets)
    tl.store(Y + offsets, x * scale)
  }

/-- Grouping both the complete RHS and its literal base remains transparent. -/
def groupedSource (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature profile in triton {
    scale: tl.constexpr = (((2.0) ** (127 - 15)))
    offsets = tl.arange(0, B)
    x = tl.load(X + offsets)
    y = x * scale
    tl.store(Y + offsets, y)
  }

theorem grouped_source (X Y : RegionName) (B : Nat) :
    groupedSource X Y B = expanded X Y B := rfl

#axiomsClean grouped_source
#axiomsClean complete_source
#axiomsClean reciprocal_source
end
end VeriTile.Tests.ConstexprPower
