import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.TypedStores
open VeriTile.Triton

def byteSignature : CallSignature where
  pointers := [("X",.u8),("Y",.u8)]

def byteCopy (X Y : RegionName) : ComputeKernel :=
  with_triton_signature byteSignature in triton {
    x = tl.load(X)
    y = x
    tl.store(Y, y)
  }

def byteReference (X Y : RegionName) : ComputeKernel := triton {
  tl.region X = tl.uint8, Y = tl.uint8
  x = tl.load(X)
  y = x
  tl.store(Y, y)
}

/-- An identity storage conversion must not introduce a new modulo operation. -/
theorem byte_identity (X Y : RegionName) : byteCopy X Y = byteReference X Y := rfl

def byteArithmetic (X Y : RegionName) : ComputeKernel :=
  with_triton_signature byteSignature in triton {
    x = tl.load(X)
    x += 1
    tl.store(Y, x)
  }

def byteArithmeticReference (X Y : RegionName) : ComputeKernel := triton {
  tl.region X = tl.uint8, Y = tl.uint8
  x = tl.load(X)
  x += 1
  tl.store(Y, x.to(tl.uint8))
}

/-- Arithmetic invalidates load provenance; the destination conversion stays. -/
theorem arithmetic_conversion (X Y : RegionName) :
    byteArithmetic X Y = byteArithmeticReference X Y := rfl

def narrowSignature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp16)]

def narrowCopy (X Y : RegionName) : ComputeKernel :=
  with_triton_signature narrowSignature in triton {
    p = Y + 1
    x = tl.load(X)
    tl.store(p, x)
  }

def narrowReference (X Y : RegionName) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float16
  p = Y + 1
  x = tl.load(X)
  tl.store(p, x.to(tl.float16))
}

/-- Pointer aliases retain destination FP16 rounding when the source is FP32. -/
theorem cross_dtype_conversion (X Y : RegionName) : narrowCopy X Y = narrowReference X Y := rfl

def overwriteSignature : CallSignature where
  pointers := [("X",.fp16),("Z",.fp32),("Y",.fp16)]

def overwritten (X Z Y : RegionName) : ComputeKernel :=
  with_triton_signature overwriteSignature in triton {
    x = tl.load(X)
    x = tl.load(Z)
    tl.store(Y, x)
  }

def overwrittenReference (X Z Y : RegionName) : ComputeKernel := triton {
  tl.ports inputs=[X,Z], outputs=[Y]
  tl.region X = tl.float16, Z = tl.float32, Y = tl.float16
  x = tl.load(X)
  x = tl.load(Z)
  tl.store(Y, x.to(tl.float16))
}

/-- Reassignment replaces, rather than retaining, the earlier source dtype. -/
theorem reassign_conversion (X Z Y : RegionName) :
    overwritten X Z Y = overwrittenReference X Z Y := rfl

def integerWhere (Y : RegionName) (B : Nat) : ComputeKernel := triton {
  i = tl.arange(0, B)
  x = tl.where(i < B, i, 0)
  tl.store(Y + i, x)
}

def integerWhereReference (Y : RegionName) (B : Nat) : ComputeKernel := triton {
  i = tl.arange(0, B)
  x = tl.where(i < B, i, $(0))
  tl.store(Y + i, x)
}

/-- Ordinary Python integer literals need no Lean antiquote to select indices. -/
theorem integer_literal (Y : RegionName) (B : Nat) :
    integerWhere Y B = integerWhereReference Y B := rfl

def castLoadSignature : CallSignature where
  pointers := [("X",.i64),("Y",.i64)]

def castLoad (X Y : RegionName) : ComputeKernel :=
  with_triton_signature castLoadSignature in triton {
    x = tl.load(X).to(tl.int64)
    tl.store(Y, x)
  }

/-- The load inside the parser's combined load/cast syntax is a real input. -/
theorem cast_load_port (X Y : RegionName) : (castLoad X Y).inputs = [X] := rfl

/-- error: pointer parameter X is missing from the call signature -/
#guard_msgs in
def missingCastInput (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("Y",.fp32)]} : CallSignature) in triton {
    x = tl.load(X).to(tl.float32)
    tl.store(Y, x)
  }

/-- error: Lean antiquotes do not belong in the pasted body -/
#guard_msgs in
def hiddenAntiquote (Y : RegionName) : ComputeKernel :=
  with_triton_signature byteSignature in triton {
    x = $(0)
    tl.store(Y, x)
  }

/-- Integer loop reassignment must not reuse entry-width provenance for a
later store, including when the loop executes zero times. -/
def integerLoop (Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature ({pointers := [("Y",.i8)], constexpr := ["B"]} : CallSignature) in triton {
    x = tl.full([], 0, tl.int64)
    for i in range(B):
      x = x + i
    tl.store(Y, x)
  }

def integerLoopReference (Y : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region Y = tl.int8
  x = tl.full([], 0, tl.int64)
  tl.for i in $(B) {
    x = x + i
  }
  tl.store(Y, x.to(tl.int8))
}

theorem integer_loop_conversion (Y : RegionName) (B : Nat) :
    integerLoop Y B = integerLoopReference Y B := rfl

/-- error: loop-carried value requires a stable, known source dtype -/
#guard_msgs in
def changingFloatLoop (X Z Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.fp32),("Z",.fp16),("Y",.fp32)], constexpr := ["B"]} : CallSignature) in triton {
    x = tl.load(X)
    for i in range(B):
      x = tl.load(Z)
    tl.store(Y, x)
  }

#axiomsClean integer_loop_conversion
#axiomsClean byte_identity
#axiomsClean arithmetic_conversion
#axiomsClean cross_dtype_conversion
#axiomsClean reassign_conversion
#axiomsClean integer_literal
#axiomsClean cast_load_port
end VeriTile.Tests.TypedStores
