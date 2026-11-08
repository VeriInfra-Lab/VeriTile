import VeriTile.Triton.DSL.BinaryPowerConstants
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.BinaryPowerConstants
open VeriTile.Triton
noncomputable section

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("n",.i64)]
  constexpr := ["B"]
  readPorts := ["X"]
  writePorts := ["Y"]

def copied (X Y : RegionName) (n : Int) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [n := n] in triton {
    scale: tl.constexpr = 2.0 ** (127 - 15)
    offsets = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B).to(tl.int64)
    x = tl.load(X + offsets, mask=offsets < n)
    tl.store(Y + offsets, x * scale, mask=offsets < n)
  }

/-- Compile-time evaluation changes no runtime instruction, signed cast, mask,
pointer, source precision or arithmetic order, for any invocation. -/
theorem complete_program (X Y : RegionName) (n : Int) (B : Nat) :
    copied X Y n B =
      (let scale : ℝ := 2^112
       with_triton_call signature call_args [n := n] in triton {
         offsets = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B).to(tl.int64)
         x = tl.load(X + offsets, mask=offsets < n)
         tl.store(Y + offsets, x * $(scale), mask=offsets < n)
       }) := rfl

def captured (X Y : RegionName) (scale : Int) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [n := scale] in triton {
    scale: tl.constexpr = 2.0 ** (3 * 4)
    x = tl.load(X, mask=n > 0)
    tl.store(Y, x * scale, mask=n > 0)
  }

/-- A source-local constant never captures an identically named caller argument. -/
theorem caller_argument (X Y : RegionName) (scale : Int) (B : Nat) :
    captured X Y scale B =
      (let value : ℝ := 2^12
       with_triton_call signature call_args [n := scale] in triton {
         x = tl.load(X, mask=n > 0)
         tl.store(Y, x * $(value), mask=n > 0)
       }) := rfl

/-- error: exact FP32 binary-power constexpr requires exponent in [0,127] -/
#guard_msgs in
def overflow (X Y : RegionName) (n : Int) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [n := n] in triton {
    scale: tl.constexpr = 2.0 ** 128
    tl.store(Y, scale)
  }

/-- error: binary-power exponent must be a closed integer expression -/
#guard_msgs in
def runtime (X Y : RegionName) (n : Int) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [n := n] in triton {
    scale: tl.constexpr = 2.0 ** n
    tl.store(Y, scale)
  }

/-- error: a floating constexpr cannot be reassigned -/
#guard_msgs in
def reassigned (X Y : RegionName) (n : Int) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [n := n] in triton {
    scale: tl.constexpr = 2.0 ** 7
    scale = 3.0
    tl.store(Y, scale)
  }

#axiomsClean complete_program
#axiomsClean caller_argument
end
end VeriTile.Tests.BinaryPowerConstants
