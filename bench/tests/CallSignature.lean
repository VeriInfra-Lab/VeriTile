import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Semantics.Step
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.CallSignature
open VeriTile.Triton
noncomputable section

def signature : Triton.CallSignature where
  pointers := [("X",.i8),("Y",.fp16)]
  scalars := [("gain",.fp32)]
  constexpr := ["B"]

def pasted (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in
  triton {
    offsets = tl.arange(0, B).to(tl.int64)
    x = tl.load(X + offsets)
    y = x * gain
    tl.store(Y + offsets, y, mask=offsets < B)
  }

def expanded (X Y : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region X = tl.int8, Y = tl.float16
  tl.param gain : tl.float32
  offsets = tl.arange(0, B).to(tl.int64)
  x = tl.load(X + offsets)
  y = x * gain
  tl.store(Y + offsets, (y).to(tl.float16), mask=offsets < B)
}

/-- The input stays signed INT8 and the masked output retains FP16 rounding. -/
theorem implicit_store_and_runtime_input (X Y : RegionName) (B : Nat) :
    pasted X Y B = expanded X Y B := rfl

def gainSignature : Triton.CallSignature where
  pointers := [("Y",.fp32)]
  scalars := [("gain",.fp32)]

def storeGain (Y : RegionName) : ComputeKernel :=
  with_triton_signature gainSignature in
  triton {
    tl.store(Y, gain)
  }

/-- The same pasted body accepts different call values without rewriting it. -/
theorem actual_argument (Y : RegionName) (gain : ℝ) (s : BlockState) :
    (exec ((storeGain Y).withScalarArguments [ScalarArgument.fp32 "gain" gain]).eraseDType.toAlgKernel s).map
      (fun t => t.readMem Y 0) = some gain := by
  simp [storeGain,ComputeKernel.withScalarArguments,ScalarArgument.fp32,ScalarArgument.statement,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,ComputeKernel.toAlgKernel,
    ComputeKernel.toAlgorithm?,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,ComputeKernel.fromKernelBody,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,ComputeDType.eraseDType,
    VeriTile.Triton.eraseDType,Except.bind,bind,eraseDType_float,
    exec,stepStmts,stepStmt,evalOp.eq_def,Tile.bop,Tile.cop]

/-- error: pointer parameter X is missing from the call signature -/
#guard_msgs in
def missingPointer (X : RegionName) : ComputeKernel :=
  with_triton_signature ({} : Triton.CallSignature) in
  triton { x = tl.load(X) }

/-- error: duplicate name in call signature -/
#guard_msgs in
def duplicate : ComputeKernel :=
  with_triton_signature ({scalars := [("gain",.fp32),("gain",.fp64)]} : Triton.CallSignature) in
  triton { y = gain }

/-- error: pointer types belong in the call signature, outside the pasted body -/
#guard_msgs in
def bodyMetadata (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in
  triton {
    tl.region X = tl.int8
    x = tl.load(X)
  }

/-- error: pointer parameter p is missing from the call signature -/
#guard_msgs in
def escapedLoopAlias (Y : RegionName) : ComputeKernel :=
  with_triton_signature gainSignature in
  triton {
    for i in range(4) {
      p = Y + i
    }
    tl.store(p, gain)
  }

/-- error: cannot determine tl.store destination dtype from the call signature -/
#guard_msgs in
def unsupportedPointerSelection (Y : RegionName) : ComputeKernel :=
  with_triton_signature gainSignature in
  triton {
    tl.store(tl.where(gain > 0, Y, Y), gain)
  }

#axiomsClean implicit_store_and_runtime_input
#axiomsClean actual_argument
end
/-- Immutable constexpr division is a call-time constant, even for an odd size. -/
def staticQuotient (N : Nat) : ComputeKernel :=
  with_triton_signature ({constexpr := ["N"]} : CallSignature) in triton {
    half = N // 2
  }

theorem static_quotient (N : Nat) : staticQuotient N = (triton {
  half = $((N / 2 : Nat))
} : ComputeKernel) := rfl

/-- Supplying a runtime call value must not turn its division into constexpr. -/
def runtimeQuotient (N : Nat) : ComputeKernel :=
  with_triton_call ({scalars := [("N",.integer)]} : CallSignature) call_args [N := N] in triton {
    half = N // 2
  }

theorem runtime_quotient (N : Nat) : runtimeQuotient N = (triton {
  half = $((N : Nat)) // $((2 : Nat))
} : ComputeKernel) := rfl

/-- An overwritten constexpr name refers to the local register. -/
def mutableQuotient (N : Nat) : ComputeKernel :=
  with_triton_signature ({constexpr := ["N"]} : CallSignature) in triton {
    N = tl.cast(5, tl.int64)
    half = N // 2
  }

theorem mutable_quotient (N : Nat) : mutableQuotient N = (triton {
  N = tl.cast(5, tl.int64)
  half = N // $((2 : Int))
} : ComputeKernel) := rfl

#axiomsClean static_quotient
#axiomsClean runtime_quotient
#axiomsClean mutable_quotient
/-- A genuinely compile-time nonzero tile width may divide another constexpr. -/
def constexprWidthDivisor (N widthMinusOne : Nat) : ComputeKernel :=
  let B := widthMinusOne + 1
  with_triton_signature ({constexpr := ["N","B"]} : CallSignature) in triton {
    count = N // B
  }
theorem constexpr_width_quotient (N widthMinusOne : Nat) :
    constexprWidthDivisor N widthMinusOne = triton {
      count = $(N / (widthMinusOne + 1))
    } := rfl

/-- Literal division by zero must retain its original integer operation. -/
def zeroDivisor (N : Nat) : ComputeKernel :=
  with_triton_signature ({constexpr := ["N"]} : CallSignature) in triton {
    count = N // 0
  }
theorem zero_divisor_not_folded (N : Nat) : zeroDivisor N = triton {
    count = $(N) // $((0 : Nat))
  } := rfl

#axiomsClean constexpr_width_quotient
#axiomsClean zero_divisor_not_folded

/-- A pasted compile-time split loop keeps the original quotient and arange. -/
def tiledSignature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  constexpr := ["N","B"]
def tiledStaticRange (X Y : RegionName) (N widthMinusOne : Nat) : ComputeKernel :=
  let B := widthMinusOne + 1
  with_triton_signature tiledSignature in triton {
    for q in tl.static_range(N // B):
      offsets = tl.arange(q * B, (q + 1) * B)
      x = tl.load(X + offsets)
      tl.store(Y + offsets, x)
  }
theorem tiled_static_range (X Y : RegionName) (N widthMinusOne : Nat) :
    tiledStaticRange X Y N widthMinusOne = (triton {
      tl.region X = tl.float32, Y = tl.float32
      for q in tl.static_range($(N / (widthMinusOne + 1))) {
        offsets = q * $(widthMinusOne + 1) + tl.arange(0, $(widthMinusOne + 1))
        x = tl.load(X + offsets)
        tl.store(Y + offsets, x)
      }
    } : ComputeKernel) := rfl

/-- error: tl.static_range requires an immutable Nat constexpr bound -/
#guard_msgs in
def runtimeStaticRange (N : Nat) : ComputeKernel :=
  with_triton_call ({scalars := [("N",.integer)]} : CallSignature) call_args [N := N] in triton {
    for i in tl.static_range(N):
      x = i
  }

/-- error: tl.static_range requires an immutable Nat constexpr bound -/
#guard_msgs in
def mutableStaticRange (N : Nat) : ComputeKernel :=
  with_triton_signature ({constexpr := ["N"]} : CallSignature) in triton {
    N = tl.cast(4, tl.int64)
    for i in tl.static_range(N):
      x = i
  }

#axiomsClean tiled_static_range
/-- Source integer literals retain signed metadata arithmetic and comparisons. -/
def signedSourceLiterals (I : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("I",.i32)]} : CallSignature) in triton {
    length = tl.load(I) - 1
    head = tl.program_id(0).to(tl.int64)
    leader = head == 0
    shifted = head - 1
    twice = 2 * shifted
  }
theorem signed_source_literals (I : RegionName) :
    signedSourceLiterals I = (triton {
      tl.region I = tl.int32
      length = tl.load(I) - $((1 : Int))
      head = tl.program_id(0).to(tl.int64)
      leader = head == $((0 : Int))
      shifted = head - $((1 : Int))
      twice = $((2 : Int)) * shifted
    } : ComputeKernel) := rfl

/-- error: arithmetic: dtype mismatch -/
#guard_msgs in
def floatLiteralIsNotInteger (I : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("I",.i32)]} : CallSignature) in triton {
    length = tl.load(I) - 1.0
  }
#axiomsClean signed_source_literals
/-- Entry arguments are used before their assignment; later reads use locals. -/
def overwrittenIntegers (n m : Int) : ComputeKernel :=
  with_triton_call ({scalars := [("n",.integer),("m",.integer)]} : CallSignature)
    call_args [n := n, m := m] in triton {
      before = n
      n = tl.cast(n, tl.int64)
      m = tl.cast(m, tl.int64)
      n = n + m
      after = n
    }
theorem integer_assignment_order (n m : Int) : overwrittenIntegers n m = triton {
    before = $((n : Int))
    n = tl.cast($((n : Int)), tl.int64)
    m = tl.cast($((m : Int)), tl.int64)
    n = n + m
    after = n
  } := rfl

def overwrittenLoop (n : Int) (trips : Nat) : ComputeKernel :=
  with_triton_call ({scalars := [("n",.integer),("trips",.integer)]} : CallSignature)
    call_args [n := n, trips := trips] in triton {
      n = tl.cast(n, tl.int64)
      for i in range(trips):
        n += 1
      after = n
    }
theorem integer_assignment_loop (n : Int) (trips : Nat) : overwrittenLoop n trips = triton {
    n = tl.cast($((n : Int)), tl.int64)
    tl.for i in $(trips) {
      n += $((1 : Int))
    }
    after = n
  } := rfl

def conditionalArgumentOverwrite (n : Int) (flag : Bool) : ComputeKernel :=
  with_triton_call ({scalars := [("n",.integer)], constexpr := ["flag"]} : CallSignature)
    call_args [n := n] in triton {
      if flag:
        n = tl.cast(n, tl.int64)
      after = n
    }

/-- Each branch binds the remaining source independently; the unassigned path
reads the original signed input and the assigned path reads the local cast. -/
theorem conditional_argument_binding (n : Int) (flag : Bool) :
    conditionalArgumentOverwrite n flag = triton {
      if $((flag : Bool)) {
        n = tl.cast($((n : Int)), tl.int64)
        after = n
      } else {
        after = $((n : Int))
      }
    } := rfl
#axiomsClean conditional_argument_binding

/-- error: per-call specialization does not yet support reassigned scalar arguments -/
#guard_msgs in
def loopArgumentOverwrite (n : Int) (trips : Nat) : ComputeKernel :=
  with_triton_call ({scalars := [("n",.integer),("trips",.integer)]} : CallSignature)
    call_args [n := n, trips := trips] in triton {
      for i in range(trips):
        n = tl.cast(n, tl.int64)
      after = n
    }

/-- error: per-call specialization does not yet support reassigned scalar arguments -/
#guard_msgs in
def floatingArgumentOverwrite (gain : ℝ) : ComputeKernel :=
  with_triton_call ({scalars := [("gain",.fp32)]} : CallSignature)
    call_args [gain := gain] in triton {
      gain = gain + 1.0
    }


/-- Even an empty source loop sees the initialized argument, independent of
arbitrary stale registers in the caller state. -/
theorem integer_assignment_zero_trip {A : Type} [Inhabited A]
    (M : FP.Structural.Algebra A) (n : Int) (s : FP.Structural.State A) :
    (FP.Structural.exec M (overwrittenLoop n 0) s).map (fun t => t.regs .int [] "after") =
      some (some (fun _ => n)) := by
  simp [integer_assignment_loop,FP.Structural.exec,FP.Structural.run,FP.Structural.step,
    FP.Structural.loop,FP.Structural.evalExpr,FP.Structural.evalOp_unfold,FP.Structural.State.setReg]

/-- The next source read observes the updated local, preserving integer dtype. -/
theorem integer_assignment_one_trip {A : Type} [Inhabited A]
    (M : FP.Structural.Algebra A) (n : Int) (s : FP.Structural.State A) :
    (FP.Structural.exec M (overwrittenLoop n 1) s).map (fun t => t.regs .int [] "after") =
      some (some (fun _ => n + 1)) := by
  simp [integer_assignment_loop,FP.Structural.exec,FP.Structural.run,FP.Structural.step,
    FP.Structural.loop,FP.Structural.evalExpr,FP.Structural.evalOp_unfold,FP.Structural.State.setReg,
    FP.Structural.numeric,FP.Structural.bop]
  rfl
#axiomsClean integer_assignment_zero_trip
#axiomsClean integer_assignment_one_trip

#axiomsClean integer_assignment_order
#axiomsClean integer_assignment_loop

end VeriTile.Tests.CallSignature

namespace VeriTile.Tests.CallSignature
open VeriTile.Triton

/-- An unresolved legacy antiquote is not evidence of an integer operand. -/
example (p : ℝ) : (triton {
    a = 1 + $(p)
    b = 1 - $(p)
    c = 1 * $(p)
  }) = (triton {
    a = $((1 : ℝ)) + $(p)
    b = $((1 : ℝ)) - $(p)
    c = $((1 : ℝ)) * $(p)
  }) := rfl

/-- Explicit caller integer carriers still determine the plain literal. -/
example (n : Nat) : (triton { a = 1 + $((n : Nat)) }) =
    (triton { a = $((1 : Nat)) + $((n : Nat)) }) := rfl
example (n : Int) : (triton { a = 1 + $((n : Int)) }) =
    (triton { a = $((1 : Int)) + $((n : Int)) }) := rfl
end VeriTile.Tests.CallSignature


namespace VeriTile.Tests.CallSignature.PointerCastStore
open VeriTile.Triton
noncomputable section

def profile : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("stride",.i32)]

-- The explicit source cast remains; storing its FP32 result must add no second
-- cast, including after rebinding the typed output pointer.
def attributeCast (X Y : RegionName) (n : Nat) : ComputeKernel :=
  with_triton_call profile call_args [stride := n] in
  triton {
    Y += stride
    x = tl.load(X)
    y = (x * x).to(Y.dtype.element_ty)
    output = y * x
    tl.store(Y, output)
  }
def explicitCast (X Y : RegionName) (n : Nat) : ComputeKernel :=
  with_triton_call profile call_args [stride := n] in
  triton {
    Y += stride
    x = tl.load(X)
    y = (x * x).to(tl.float32)
    output = y * x
    tl.store(Y, output)
  }
example (X Y : RegionName) (n : Nat) : attributeCast X Y n = explicitCast X Y n := rfl

def legacyAttribute (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.fp32),("Y",.fp16)]} : CallSignature) in
  triton {
    x = tl.load(X)
    y = x.to(Y.type.element_ty)
    tl.store(Y, y)
  }
def explicitHalf (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.fp32),("Y",.fp16)]} : CallSignature) in
  triton {
    x = tl.load(X)
    y = x.to(tl.float16)
    tl.store(Y, y)
  }
example (X Y : RegionName) : legacyAttribute X Y = explicitHalf X Y := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.CallSignature.PointerCastStore

namespace VeriTile.Tests.CallSignature.IntegerCastLiterals
open VeriTile.Triton

def signature : CallSignature where
  pointers := []
  scalars := []
  constexpr := []

def source : ComputeKernel :=
  with_triton_signature signature in
  triton {
    total = tl.cast(0, tl.int64)
    for i in range(tl.cast(-2, tl.int64), tl.cast(1, tl.int64)):
        total += i
  }

/-- The three signed iterations retain -2,-1,0 for every opaque FP backend.
No floating-to-integer callback is needed for Python integer literal casts. -/
theorem execution {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (s : FP.Structural.State α) :
    (FP.Structural.exec M source s).map (fun t => t.regs .int [] "total") =
      some (some (fun _ => (-3 : Int))) := by
  simp only [source, FP.Structural.exec]
  iterate 8
    all_goals try simp only [FP.Structural.run, FP.Structural.step,
      FP.Structural.evalExpr, FP.Structural.evalOp]
    all_goals try simp [FP.Structural.State.setReg, FP.Structural.numeric,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals try rw [FP.Structural.rangeInt]
  rfl
#axiomsClean execution
end VeriTile.Tests.CallSignature.IntegerCastLiterals

namespace VeriTile.Tests.CallSignature.PositionalMaximumStore
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := []
  constexpr := ["B"]

def positional (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in
  triton {
    lanes = tl.arange(0, B)
    values = tl.load(X + lanes)
    maximum = tl.max(values, 0)
    tl.store(Y, maximum)
  }
def named (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in
  triton {
    lanes = tl.arange(0, B)
    values = tl.load(X + lanes)
    maximum = tl.max(values, axis=0)
    tl.store(Y, maximum)
  }

/-- Positional-axis syntax retains the same reduction and already-FP32 store;
there is no additional cast event for an opaque backend to interpret. -/
theorem complete_program (X Y : RegionName) (B : Nat) :
    positional X Y B = named X Y B := rfl
#axiomsClean complete_program
end VeriTile.Tests.CallSignature.PositionalMaximumStore

namespace VeriTile.Tests.CallSignature.MixedFP32Storage
open VeriTile.Triton
noncomputable section

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := []
  constexpr := []
  typedFP32Pointers := ["X"]

def source (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(X).to(tl.float32)
    y = tl.load(Y).to(tl.float32)
  }
/-- The two source-FP32 pointers retain distinct interpreter cell channels
required by the existing IO. Both explicit source conversion events remain. -/
theorem complete_channels (X Y : RegionName) : source X Y =
    (let dtype : FloatDType := .fp32; triton {
      tl.ports inputs=[X,Y], outputs=[]
      tl.region X = DTYPE, Y = tl.float32
      x = tl.load(X).to(tl.float32)
      y = tl.load(Y).to(tl.float32)
    }) := rfl

/-- error: typed FP32 pointers must be distinct FP32 pointer parameter names -/
#guard_msgs in
def unknownPointer (X : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.fp32)], typedFP32Pointers := ["Z"]} : CallSignature) in
  triton { x = tl.load(X) }

/-- error: typed FP32 pointers must be distinct FP32 pointer parameter names -/
#guard_msgs in
def wrongSourceDtype (X : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.fp16)], typedFP32Pointers := ["X"]} : CallSignature) in
  triton { x = tl.load(X) }

/-- error: typed FP32 pointers must be distinct FP32 pointer parameter names -/
#guard_msgs in
def duplicatePointer (X : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.fp32)], typedFP32Pointers := ["X","X"]} : CallSignature) in
  triton { x = tl.load(X) }

#axiomsClean complete_channels
end
end VeriTile.Tests.CallSignature.MixedFP32Storage

namespace VeriTile.Tests.CallSignature.ScanStorage
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32),("Half",.fp16)]
  scalars := []
  constexpr := ["B"]

def source (X Y Half : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    lanes = tl.arange(0, B)
    value = tl.load(X + lanes).to(tl.float32)
    inclusive = tl.cumsum(value, axis=0)
    tl.store(Y + lanes, inclusive)
    tl.store(Half + lanes, inclusive)
  }

/-- An FP32 scan needs no same-dtype store conversion. An FP16 destination
still requires its actual narrowing operation. Both stores are checked together. -/
theorem complete_program (X Y Half : RegionName) (B : Nat) :
    source X Y Half B = triton {
      tl.ports inputs=[X], outputs=[Y,Half]
      tl.region X = tl.float32, Y = tl.float32, Half = tl.float16
      lanes = tl.arange(0, B)
      value = tl.load(X + lanes).to(tl.float32)
      inclusive = tl.cumsum(value, axis=0)
      tl.store(Y + lanes, inclusive)
      tl.store(Half + lanes, inclusive.to(tl.float16))
    } := rfl

#axiomsClean complete_program
end VeriTile.Tests.CallSignature.ScanStorage

namespace VeriTile.Tests.CallSignature.LiveDotStorage
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32),("Z",.fp32)]
  constexpr := ["B"]

def source (X Y Z : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    offsets = tl.arange(0, B)[:, None] * B + tl.arange(0, B)[None, :]
    a = tl.load(X + offsets)
    b = tl.load(Y + offsets)
    acc = tl.zeros((B, B), dtype=tl.float32)
    for k in range(2):
        acc = tl.dot(a, b, acc)
    c = tl.cast(acc, tl.float16)
    tl.store(Z + offsets, c)
  }

/-- Keep the live dot inside every iteration and both real conversion events:
FP16 narrowing followed by implicit FP32 output storage. No cast law is used. -/
theorem complete_program (X Y Z : RegionName) (B : Nat) : source X Y Z B = triton {
    tl.ports inputs=[X,Y], outputs=[Z]
    tl.region X = tl.float32, Y = tl.float32, Z = tl.float32
    offsets = tl.arange(0, B)[:, None] * B + tl.arange(0, B)[None, :]
    a = tl.load(X + offsets)
    b = tl.load(Y + offsets)
    acc = tl.zeros([B, B], dtype=tl.float32)
    tl.for k in $(2) {
      acc = tl.dot(a, b, acc)
    }
    c = tl.cast(acc, tl.float16)
    tl.store(Z + offsets, c.to(tl.float32))
  } := rfl

/-- error: loop-carried value requires a stable, known source dtype -/
#guard_msgs in
def invalidHalfCarry (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.fp32),("Y",.fp32)], constexpr := ["B"]} : CallSignature) in triton {
    offsets = tl.arange(0, B)[:, None] * B + tl.arange(0, B)[None, :]
    a = tl.load(X + offsets)
    b = tl.load(Y + offsets)
    acc = tl.zeros((B, B), dtype=tl.float16)
    for k in range(2):
        acc = tl.dot(a, b, acc)
  }

#axiomsClean complete_program
end VeriTile.Tests.CallSignature.LiveDotStorage


namespace VeriTile.Tests.CallSignature.StringChoice
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  constexpr := ["mode"]

def source (X Y : RegionName) (chooseLeft : Bool) : ComputeKernel :=
  let mode := if chooseLeft then "left" else "right"
  with_triton_signature signature in triton {
    x = tl.load(X)
    if mode != "left":
        result = -x
    elif mode == "left":
        result = x
    tl.store(Y, result)
  }

/-- A String if/elif is exhaustive only because the actual caller supplies
these two literal alternatives. The redundant second test adds no cast. -/
theorem complete_program (X Y : RegionName) (chooseLeft : Bool) :
    source X Y chooseLeft =
      (let other := !chooseLeft; triton {
        tl.ports inputs=[X], outputs=[Y]
        tl.region X = tl.float32, Y = tl.float32
        x = tl.load(X)
        if other {
          result = -x
        } else {
          result = x
        }
        tl.store(Y, result)
      }) := by cases chooseLeft <;> rfl

/-- error: string constexpr requires a literal or a finite conditional of string literals -/
#guard_msgs in
def opaqueString (mode : String) : ComputeKernel :=
  with_triton_signature ({constexpr := ["mode"]} : CallSignature) in triton {
    if mode == "left":
        result = 1
  }

/-- error: string constexpr mutation is unsupported -/
#guard_msgs in
def mutatedString : ComputeKernel :=
  let mode := "left"
  with_triton_signature ({constexpr := ["mode"]} : CallSignature) in triton {
    mode = "right"
  }

#axiomsClean complete_program
end VeriTile.Tests.CallSignature.StringChoice


namespace VeriTile.Tests.CallSignature.UninferredNumericalCaller
open VeriTile.Triton

def source (X : RegionName) : ComputeKernel :=
  let B := 64
  with_triton_signature ({pointers := [("X",.fp32)], constexpr := ["B"]} : CallSignature) in
  triton {
    lanes = tl.arange(0, B)
    value = tl.load(X + lanes)
    tl.store(X + lanes, value)
  }

/-- Inspecting whether a caller is a String must never assign String to an
unsolved numeral type. The pre-existing Nat specialization still determines B. -/
theorem complete_program (X : RegionName) : source X = triton {
    tl.ports inputs=[X], outputs=[X]
    tl.region X = tl.float32
    lanes = tl.arange(0, $(64))
    value = tl.load(X + lanes)
    tl.store(X + lanes, value)
  } := rfl
#axiomsClean complete_program
end VeriTile.Tests.CallSignature.UninferredNumericalCaller

namespace VeriTile.Tests.CallSignature.IndexedMaximumStorage
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("V",.fp32),("I",.i64)]
  constexpr := ["B"]

def source (X V I : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    lanes = tl.arange(0, B)
    values = tl.load(X + lanes)
    maximum, position = tl.max(values, axis=0, return_indices=True)
    tl.store(V, maximum)
    tl.store(I, position)
  }

/-- A paired FP32 reduction already returns an FP32 value. The index still
needs the actual signed64 output conversion; no FP cast identity is assumed. -/
theorem complete_program (X V I : RegionName) (B : Nat) : source X V I B = triton {
    tl.ports inputs=[X], outputs=[V,I]
    tl.region X = tl.float32, V = tl.float32, I = tl.int64
    lanes = tl.arange(0, B)
    values = tl.load(X + lanes)
    maximum, position = tl.max(values, axis=0, return_indices=True)
    tl.store(V, maximum)
    tl.store(I, position.to(tl.int64))
  } := rfl
#axiomsClean complete_program
end VeriTile.Tests.CallSignature.IndexedMaximumStorage

namespace VeriTile.Tests.CallSignature.NaturalIntegerMemory
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("P",.i64),("Y",.fp32)]
  constexpr := ["B"]
  naturalIntegerPointers := ["P"]

def source (X Y : RegionName) (P : Region .nat) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    lanes = tl.arange(0, B)
    values = tl.load(X + lanes)
    prefix_value = tl.load(P + lanes)
    indices = prefix_value - 1
    tl.store(Y + indices, values)
  }

/-- Source integer storage can use an explicitly chosen existing natural IO
carrier. The complete AST retains that channel and integer arithmetic; it
neither inserts an FP32 store cast nor claims a native integer range bound. -/
theorem complete_program (X Y : RegionName) (P : Region .nat) (B : Nat) :
    source X Y P B = ComputeKernel.mk [X,P.cast] [Y]
      [.assign .nat [B] "lanes" (.alg (.arange B)),
       .assign .real [B] "values" (.compute (.load .fp32
         (.region X (.ref .nat [B] "lanes")) .none)),
       .assign .nat [B] "prefix_value" (.alg (.load .nat
         (.region P (.ref .nat [B] "lanes")) .none)),
       .assign .nat [B] "indices" (.alg (.sub .nat Broadcast.scalarR
         (.ref .nat [B] "prefix_value") (.constNat 1))),
       .store .real [B] (.region Y (.ref .nat [B] "indices"))
         (.compute (.alg .fp32 (.ref .real [B] "values"))) .none] := rfl

/-- error: natural integer pointer requires an already typed Region .nat caller -/
#guard_msgs in
example (X Y P : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    lanes = tl.arange(0, B)
    values = tl.load(P + lanes)
    tl.store(Y + lanes, values)
  }

/-- error: natural integer pointers must be distinct integer pointer parameter names -/
#guard_msgs in
example (X Y : RegionName) (P : Region .nat) (B : Nat) : ComputeKernel :=
  with_triton_signature { signature with naturalIntegerPointers := ["X"] } in triton {
    lanes = tl.arange(0, B)
    values = tl.load(P + lanes)
    tl.store(Y + lanes, values)
  }

/-- error: natural integer pointers must be distinct integer pointer parameter names -/
#guard_msgs in
example (X Y : RegionName) (P : Region .nat) (B : Nat) : ComputeKernel :=
  with_triton_signature { signature with naturalIntegerPointers := ["P","P"] } in triton {
    lanes = tl.arange(0, B)
    values = tl.load(P + lanes)
    tl.store(Y + lanes, values)
  }

#axiomsClean complete_program
end VeriTile.Tests.CallSignature.NaturalIntegerMemory

namespace VeriTile.Tests.PythonReservedNames
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  constexpr := ["B"]

def source (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    partial = tl.program_id(0)
    prefix = partial * B + tl.arange(0, B)
    values = tl.load(X + prefix)
    tl.store(Y + prefix, values)
  }

theorem complete_program (X Y : RegionName) (B : Nat) : source X Y B = triton {
    tl.ports inputs=[X], outputs=[Y]
    tl.region X = tl.float32, Y = tl.float32
    «partial» = tl.program_id(0)
    «prefix» = «partial» * $(B) + tl.arange(0, $(B))
    values = tl.load(X + «prefix»)
    tl.store(Y + «prefix», values)
  } := rfl
#axiomsClean complete_program
end VeriTile.Tests.PythonReservedNames

namespace VeriTile.Tests.CallSignature.InlineAddU32
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.i32),("Y",.i32),("Z",.i32)]
  constexpr := ["B"]

def source (X Y Z : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    lanes = tl.arange(0, B)
    left = tl.load(X + lanes)
    right = tl.load(Y + lanes)
    result = tl.inline_asm_elementwise("add.u32 $0, $1, $2;", constraints="=r,r,r", args=[left, right], dtype=tl.int32, is_pure=True, pack=1)
    tl.store(Z + lanes, result)
  }

/-- The complete source call compiles to the established modulo32/signed32
instruction, retaining both input reads, output storage and native ABI. -/
theorem complete_program (X Y Z : RegionName) (B : Nat) : source X Y Z B = triton {
    tl.ports inputs=[X,Y], outputs=[Z]
    tl.region X = tl.int32, Y = tl.int32, Z = tl.int32
    lanes = tl.arange(0, $(B))
    left = tl.load(X + lanes)
    right = tl.load(Y + lanes)
    result = ((left + right).to(tl.uint32)).to(tl.int32)
    tl.store(Z + lanes, result)
  } := rfl

/-- error: inline ASM: only the modeled add.u32 instruction is supported -/
#guard_msgs in
example (X Y Z : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    lanes = tl.arange(0, B)
    left = tl.load(X + lanes)
    right = tl.load(Y + lanes)
    result = tl.inline_asm_elementwise("sub.u32 $0, $1, $2;", constraints="=r,r,r", args=[left, right], dtype=tl.int32, is_pure=True, pack=1)
    tl.store(Z + lanes, result)
  }

/-- error: inline ASM: constraints must be the literal =r,r,r -/
#guard_msgs in
example (X Y Z : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    lanes = tl.arange(0, B)
    left = tl.load(X + lanes)
    right = tl.load(Y + lanes)
    result = tl.inline_asm_elementwise("add.u32 $0, $1, $2;", constraints="=r,l,l", args=[left, right], dtype=tl.int32, is_pure=True, pack=1)
    tl.store(Z + lanes, result)
  }

/-- error: inline ASM: is_pure must be literal True -/
#guard_msgs in
example (X Y Z : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    lanes = tl.arange(0, B)
    left = tl.load(X + lanes)
    right = tl.load(Y + lanes)
    result = tl.inline_asm_elementwise("add.u32 $0, $1, $2;", constraints="=r,r,r", args=[left, right], dtype=tl.int32, is_pure=False, pack=1)
    tl.store(Z + lanes, result)
  }

/-- error: inline ASM: pack must be literal 1 -/
#guard_msgs in
example (X Y Z : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    lanes = tl.arange(0, B)
    left = tl.load(X + lanes)
    right = tl.load(Y + lanes)
    result = tl.inline_asm_elementwise("add.u32 $0, $1, $2;", constraints="=r,r,r", args=[left, right], dtype=tl.int32, is_pure=True, pack=2)
    tl.store(Z + lanes, result)
  }

/-- error: inline ASM: result dtype must be tl.int32 -/
#guard_msgs in
example (X Y Z : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    lanes = tl.arange(0, B)
    left = tl.load(X + lanes)
    right = tl.load(Y + lanes)
    result = tl.inline_asm_elementwise("add.u32 $0, $1, $2;", constraints="=r,r,r", args=[left, right], dtype=tl.float32, is_pure=True, pack=1)
    tl.store(Z + lanes, result)
  }

/-- error: inline ASM: args must be two integer register identifiers -/
#guard_msgs in
example (X Y Z : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    lanes = tl.arange(0, B)
    left = tl.load(X + lanes)
    right = tl.load(Y + lanes)
    result = tl.inline_asm_elementwise("add.u32 $0, $1, $2;", constraints="=r,r,r", args=[tl.load(X + lanes), right], dtype=tl.int32, is_pure=True, pack=1)
    tl.store(Z + lanes, result)
  }

/-- error: inline ASM add.u32: operands must use integer registers -/
#guard_msgs in
example (X Y Z : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature { signature with pointers := [("X",.fp32),("Y",.fp32),("Z",.i32)] } in triton {
    lanes = tl.arange(0, B)
    left = tl.load(X + lanes)
    right = tl.load(Y + lanes)
    result = tl.inline_asm_elementwise("add.u32 $0, $1, $2;", constraints="=r,r,r", args=[left, right], dtype=tl.int32, is_pure=True, pack=1)
    tl.store(Z + lanes, result)
  }

/-- error: inline ASM: duplicate ABI option -/
#guard_msgs in
example (X Y Z : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    lanes = tl.arange(0, B)
    left = tl.load(X + lanes)
    right = tl.load(Y + lanes)
    result = tl.inline_asm_elementwise("add.u32 $0, $1, $2;", constraints="=r,r,r", args=[left, right], dtype=tl.int32, is_pure=True, pack=1, pack=1)
    tl.store(Z + lanes, result)
  }

/-- error: inline ASM: complete ABI options are required -/
#guard_msgs in
example (X Y Z : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    lanes = tl.arange(0, B)
    left = tl.load(X + lanes)
    right = tl.load(Y + lanes)
    result = tl.inline_asm_elementwise("add.u32 $0, $1, $2;", constraints="=r,r,r", args=[left, right], dtype=tl.int32, is_pure=True)
    tl.store(Z + lanes, result)
  }

/-- Python keyword order does not change the checked instruction ABI. -/
theorem reordered_abi (X Y Z : RegionName) (B : Nat) : source X Y Z B =
    with_triton_signature signature in triton {
      lanes = tl.arange(0, B)
      left = tl.load(X + lanes)
      right = tl.load(Y + lanes)
      result = tl.inline_asm_elementwise("add.u32 $0, $1, $2;", pack=1, dtype=tl.int32, args=[left, right], is_pure=True, constraints="=r,r,r")
      tl.store(Z + lanes, result)
    } := rfl

#axiomsClean reordered_abi
#axiomsClean complete_program
end VeriTile.Tests.CallSignature.InlineAddU32

namespace VeriTile.Tests.StaticAssertionSource
open VeriTile.Triton FP.Structural

def signature : CallSignature where
  pointers := [("X",.fp32)]
  constexpr := ["B"]

def source (X : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    tl.static_assert(B % 4 == 0, "B must be a multiple of four")
    lanes = tl.arange(0, B)
    value = tl.load(X + lanes)
    tl.store(X + lanes, value)
  }

theorem assertion_step_noop {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (X : RegionName) (B : Nat) :
    ((source X B).surfaceBody.head?).bind (fun statement => FP.Structural.step M statement s) = some s := by
  simp [source,ComputeKernel.surfaceBody,FP.Structural.step,FP.Structural.stepAlg,evalOp_unfold]
#axiomsClean assertion_step_noop
end VeriTile.Tests.StaticAssertionSource

namespace VeriTile.Tests.FloatingPointerFamilies
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.inputFloat),("Y",.outputFloat)]
  constexpr := ["B"]

def source (dtype out_dtype : FloatDType) (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    lanes = tl.arange(0, B)
    values = tl.load(X + lanes).to(tl.float32)
    tl.store(Y + lanes, values.to(Y.dtype.element_ty))
  }

def expected (dtype out_dtype : FloatDType) (X Y : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.ports inputs=[X], outputs=[Y]
  tl.region X = DTYPE, Y = OUT_DTYPE
  lanes = tl.arange(0, $(B))
  values = tl.load(X + lanes).to(tl.float32)
  tl.store(Y + lanes, (values).to(Y.dtype.element_ty))
}

theorem complete_kernel (dtype out_dtype : FloatDType) (X Y : RegionName) (B : Nat) :
    source dtype out_dtype X Y B = expected dtype out_dtype X Y B := rfl
#axiomsClean complete_kernel

/-- error: floating pointer family requires an already typed FloatDType caller alias -/
#guard_msgs in
example (dtype : Nat) (out_dtype : FloatDType) (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    values = tl.load(X)
    tl.store(Y, values)
  }

/-- error: inputFloat metadata cannot share the fixed typed-FP32 dtype alias -/
#guard_msgs in
example (dtype out_dtype : FloatDType) (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature { signature with typedFP32Storage := Bool.true } in triton {
    values = tl.load(X)
    tl.store(Y, values)
  }

/-- error: floating pointer family requires an already typed FloatDType caller alias -/
#guard_msgs in
example (dtype : FloatDType) (out_dtype : Nat) (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    values = tl.load(X)
    tl.store(Y, values)
  }

end VeriTile.Tests.FloatingPointerFamilies

namespace VeriTile.Tests.NaturalSignedAddress
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("P",.i32),("Out",.i64)]
  scalars := [("stride",.i64)]
  naturalIntegerPointers := ["P"]

def source (P : Region .nat) (Out : RegionName) (stride : Nat) : ComputeKernel :=
  with_triton_call signature call_args [stride := stride] in triton {
    pid = tl.program_id(0).to(tl.int64)
    index = tl.load(P + pid * stride).to(tl.int64)
    tl.store(Out + index, pid)
  }

def expected (P : Region .nat) (Out : RegionName) (stride : Nat) : ComputeKernel :=
  .mk [P.cast] [Out]
    [.assign .int [] "pid" (.alg (.castNatToInt (.programId 0))),
     .assign .int [] "index" (.alg (.castNatToInt
       (.load .nat (.region P (.castIntToNat
         (.mul .int .nil (.ref .int [] "pid") (.castNatToInt (.constNat stride))))) .none))),
     .store .int [] (.region Out (.castIntToNat (.ref .int [] "index")))
       (.alg (.ref .int [] "pid")) .none]

theorem complete_kernel (P : Region .nat) (Out : RegionName) (stride : Nat) :
    source P Out stride = expected P Out stride := rfl
#axiomsClean complete_kernel
end VeriTile.Tests.NaturalSignedAddress

namespace VeriTile.Tests.NaturalSignedOffsetRebind
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("P",.i32),("C",.i32),("Out",.u64)]
  constexpr := ["B","N"]
  naturalIntegerPointers := ["P","C"]

def source (P C Out : Region .nat) (B N : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    index = tl.load(P).to(tl.int64)
    offsets = index + tl.arange(0, B).to(tl.int64)
    counts = tl.load(C + offsets, mask=offsets < N, other=0)
    tl.store(Out + tl.arange(0, B), counts, mask=offsets < N)
  }

def expected (P C Out : Region .nat) (B N : Nat) : ComputeKernel :=
  .mk [P.cast,C.cast] [Out.cast]
    [.assign .int [] "index" (.alg (.castNatToInt (.load .nat (.region P (.constNat 0)) .none))),
     .assign .int [B] "offsets" (.alg (.add .int .scalarL (.ref .int [] "index") (.castNatToInt (.arange B)))),
     .assign .nat [B] "counts" (.alg (.load .nat
       (.region C (.castIntToNat (.ref .int [B] "offsets")))
       (.maskOther (.lt .int .scalarR (.ref .int [B] "offsets") (.castNatToInt (.constNat N)))
         (.broadcast (.constNat 0) [B])))),
     .store .nat [B] (.region Out (.arange B)) (.alg (.natToUnsigned 64 (.ref .nat [B] "counts")))
       (.mask (.lt .int .scalarR (.ref .int [B] "offsets") (.castNatToInt (.constNat N))))]

theorem complete_kernel (P C Out : Region .nat) (B N : Nat) :
    source P C Out B N = expected P C Out B N := rfl
#axiomsClean complete_kernel
end VeriTile.Tests.NaturalSignedOffsetRebind

namespace VeriTile.Tests.ImportedFiniteFunction
open VeriTile.Triton

def helper : SourceFunction :=
  triton_function isfinite_func(x) in triton {
    return _isfinited(x) if x.dtype.is_fp64() else _finitef(x.to(tl.float32))
  }

def profile32 : CallSignature where
  pointers := [("X",.fp32),("Y",.boolean)]
  functions := [helper]
  finiteImports := [("_isfinited","triton.language.extra.cuda.libdevice.isfinited"),
    ("_finitef","triton.language.extra.cuda.libdevice.finitef")]

def profile64 : CallSignature :=
  {profile32 with pointers := [("X",.fp64),("Y",.boolean)]}

def source32 (X Y : RegionName) : ComputeKernel :=
  with_triton_signature profile32 in triton {
    x = tl.load(X)
    y = isfinite_func(x)
    tl.store(Y, y)
  }

def source64 (X Y : RegionName) : ComputeKernel :=
  with_triton_signature profile64 in triton {
    x = tl.load(X)
    y = isfinite_func(x)
    tl.store(Y, y)
  }

def expected32 (X Y : RegionName) : ComputeKernel :=
  with_triton_signature {profile32 with functions := [], finiteImports := []} in triton {
    x = tl.load(X)
    y = tl.extra.cuda.libdevice.finitef(x.to(tl.float32))
    tl.store(Y, y)
  }

def expected64 (X Y : RegionName) : ComputeKernel :=
  with_triton_signature {profile64 with functions := [], finiteImports := []} in triton {
    x = tl.load(X)
    y = tl.extra.cuda.libdevice.isfinited(x)
    tl.store(Y, y)
  }

theorem single_complete (X Y : RegionName) : source32 X Y = expected32 X Y := rfl
theorem double_complete (X Y : RegionName) : source64 X Y = expected64 X Y := rfl
#axiomsClean single_complete
#axiomsClean double_complete

def wrongHelper : SourceFunction :=
  triton_function isfinite_func(x) in triton {
    return _isfinited(x) if x.dtype.is_fp64() else _finitef(x)
  }

/-- error: finite helper narrow branch requires its explicit float32 cast -/
#guard_msgs in
example (X Y : RegionName) : ComputeKernel :=
  with_triton_signature {profile32 with functions := [wrongHelper]} in triton {
    x = tl.load(X)
    y = isfinite_func(x)
    tl.store(Y, y)
  }

/-- error: finite source alias requires the explicit CUDA libdevice finitef/isfinited import -/
#guard_msgs in
example (X Y : RegionName) : ComputeKernel :=
  with_triton_signature {profile32 with finiteImports := [("_finitef","untrusted.finitef")], functions := []} in triton {
    x = tl.load(X)
    tl.store(Y, x > 0)
  }

/-- error: source import aliases must be distinct from parameters and helpers -/
#guard_msgs in
example (X Y : RegionName) : ComputeKernel :=
  with_triton_signature {profile32 with finiteImports := [("X","triton.language.extra.cuda.libdevice.finitef")]} in triton {
    x = tl.load(X)
    tl.store(Y, x > 0)
  }

/-- error: source helper and import names cannot be rebound -/
#guard_msgs in
example (X Y : RegionName) : ComputeKernel :=
  with_triton_signature profile32 in triton {
    _finitef = 0
    x = tl.load(X)
    tl.store(Y, x > 0)
  }

end VeriTile.Tests.ImportedFiniteFunction
