import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.BitValue

namespace VeriTile.Tests.ScalarAssignmentBody
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]

def ordinary (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    scale = 0.1
    x = tl.load(X)
    y = x * scale * 0.2
    tl.store(Y,y)
  }

def ordinaryExpected (X Y : RegionName) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float32
  scale = tl.full([],0.1,dtype=tl.float32)
  x = tl.load(X)
  y = x * scale * 0.2
  tl.store(Y,y)
}

/-- The assigned scalar has a precision boundary; the inline scalar remains
weak and the runtime multiplication order remains unchanged. -/
theorem ordinary_materializes (X Y : RegionName) :
    ordinary X Y = ordinaryExpected X Y := rfl

def tupleSource (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    a, b, c = 0.1, (0.2), -0.3
    x = tl.load(X)
    tl.store(Y,x * a + b * c)
  }

def tupleExpected (X Y : RegionName) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float32
  a, b, c = tl.full([],0.1,dtype=tl.float32), tl.full([],(0.2),dtype=tl.float32), tl.full([],-0.3,dtype=tl.float32)
  x = tl.load(X)
  tl.store(Y,x * a + b * c)
}

theorem tuple_materializes (X Y : RegionName) :
    tupleSource X Y = tupleExpected X Y := rfl

def zeroAndInfinity (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    zero = 0.0
    lower = float("-inf")
    upper = float("inf")
    x = tl.load(X)
    tl.store(Y,tl.minimum(tl.maximum(x + zero,lower),upper))
  }

def zeroAndInfinityExpected (X Y : RegionName) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float32
  zero = tl.full([],0.0,dtype=tl.float32)
  lower = float("-inf")
  upper = tl.bitcast(tl.full([],2139095040,dtype=tl.uint32),tl.float32)
  x = tl.load(X)
  tl.store(Y,tl.minimum(tl.maximum(x + zero,lower),upper))
}

theorem zero_and_infinity_materialize (X Y : RegionName) :
    zeroAndInfinity X Y = zeroAndInfinityExpected X Y := rfl

def signedZero (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    zero = -(0.0)
    x = tl.load(X)
    tl.store(Y,x + zero)
  }

def signedZeroExpected (X Y : RegionName) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float32
  zero = tl.bitcast(tl.full([],2147483648,dtype=tl.uint32),tl.float32)
  x = tl.load(X)
  tl.store(Y,x + zero)
}

/-- Exact negative zero must survive ordinary scalar materialization. -/
theorem signed_zero_preserved (X Y : RegionName) :
    signedZero X Y = signedZeroExpected X Y := rfl

/-- error: fp64 scalar literal assignment is not supported by this source frontend -/
#guard_msgs in
def subnormalPythonFloat (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    tiny = 1e-40
    tl.store(Y,tiny)
  }

/-- error: scalar literal assignment needs unsupported binary64-to-FP32 double rounding -/
#guard_msgs in
def doubleRounding (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    halfway = 1.000000059604644776257986737988403547205962240695953369140625
    tl.store(Y,halfway)
  }

/-- A finite decimal slightly above an FP32 midpoint rounds to that exact
midpoint in binary64. Direct FP32 rounding would produce a different word. -/
theorem double_rounding_changes_word :
    (FP.round .fp32 .ieeeValue (mkRat (2^60+2^36+1) (2^60))).raw = 0x3f800001 ∧
    (FP.round .fp32 .ieeeValue (mkRat (2^24+1) (2^24))).raw = 0x3f800000 := by
  decide +kernel

#axiomsClean ordinary_materializes
#axiomsClean tuple_materializes
#axiomsClean zero_and_infinity_materialize
#axiomsClean double_rounding_changes_word
#axiomsClean signed_zero_preserved
end VeriTile.Tests.ScalarAssignmentBody
