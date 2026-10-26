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
/-- Signed metadata may be loaded and widened inside an integer expression;
there is no floating load event to erase. -/
def nestedIntegerRead (X : RegionName) (delta : Int) : ComputeKernel :=
  with_triton_call ({pointers := [("X",.i32)],scalars := [("delta",.integer)]} : CallSignature)
    call_args [delta := delta] in triton {
    shifted = tl.load(X).to(tl.int64) + delta
  }
def nestedIntegerReference (X : RegionName) (delta : Int) : ComputeKernel := triton {
  tl.ports inputs=[X], outputs=[]
  tl.region X = tl.int32
  shifted = tl.load(X).to(tl.int64) + $((delta : Int))
}
theorem nested_integer_read_preserves_program (X : RegionName) (delta : Int) :
    nestedIntegerRead X delta = nestedIntegerReference X delta := rfl

/-- error: nested load source precision is unsupported by the current frontend -/
#guard_msgs in
def floatingReadCastToInteger (X : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.fp32)]} : CallSignature) in triton {
    shifted = tl.load(X).to(tl.int64) + 1
  }

/-- error: nested load source precision is unsupported by the current frontend -/
#guard_msgs in
def floatingReadInIntegerMask (X F : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.i32),("F",.fp32)]} : CallSignature) in triton {
    value = tl.load(X, mask=tl.load(F) > 0.0).to(tl.int64)
  }

#axiomsClean nested_integer_read_preserves_program


/-- Source dtype attributes survive constructor, broadcast, reduction, and a
possibly empty loop. The opaque backend must see no invented store cast. -/
def attributedLoopSignature : CallSignature where
  pointers := [("A",.fp32),("B",.fp32),("Y",.fp32)]
  constexpr := ["N","K"]

def attributedLoop (A B Y : RegionName) (N K : Nat) : ComputeKernel :=
  with_triton_signature attributedLoopSignature in triton {
    total = tl.zeros([N], dtype=A.dtype.element_ty)
    for k in range(K):
      a = tl.load(A + tl.arange(0, N))
      b = tl.load(B + tl.arange(0, N))
      expanded_a, _ = tl.broadcast(a, b)
      total += expanded_a * b
    tl.store(Y + tl.arange(0, N), total)
  }

theorem attribute_broadcast_loop_preserves_casts (A B Y : RegionName) (N K : Nat) :
    (attributedLoop A B Y N K).surfaceBody.getLast? =
      some (.store .real [N] (.region Y (.arange N))
        (.alg (.ref .real [N] "total")) .none) := rfl

/-- Broadcasting each result retains its own dtype, including different input
precisions and a subsequent explicit narrowing conversion. -/
def mixedBroadcast (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.fp16),("Y",.fp32)]} : CallSignature) in triton {
    x = tl.load(X)
    y = tl.load(Y)
    a, b = tl.broadcast(x, y)
    tl.store(X, a)
    tl.store(Y, b)
    z = b.to(tl.float16)
    tl.store(X, z)
    tl.store(Y, z)
  }

def mixedBroadcastReference (X Y : RegionName) : ComputeKernel := triton {
  tl.ports inputs=[X,Y], outputs=[X,Y]
  tl.region X = tl.float16, Y = tl.float32
  x = tl.load(X)
  y = tl.load(Y)
  a, b = tl.broadcast(x, y)
  tl.store(X, a)
  tl.store(Y, b)
  z = b.to(tl.float16)
  tl.store(X, z)
  tl.store(Y, z.to(tl.float32))
}

theorem broadcast_retains_each_precision (X Y : RegionName) :
    mixedBroadcast X Y = mixedBroadcastReference X Y := rfl

/-- Inferring a constructor's input dtype must still retain a real change in
the output dtype. This covers tl.full as well as the zero constructor. -/
def attributedFullSignature : CallSignature where
  pointers := [("X",.fp16),("Y",.fp32)]
  constexpr := ["N"]

def attributedFull (X Y : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_signature attributedFullSignature in triton {
    x = tl.full([N], 0, dtype=X.dtype.element_ty)
    tl.store(Y + tl.arange(0, N), x)
  }

def attributedFullReference (X Y : RegionName) (N : Nat) : ComputeKernel := triton {
  tl.ports inputs=[], outputs=[Y]
  tl.region X = tl.float16, Y = tl.float32
  x = tl.full([$(N)], 0, dtype=X.dtype.element_ty)
  tl.store(Y + tl.arange(0, $(N)), x.to(tl.float32))
}

theorem attributed_constructor_keeps_output_conversion (X Y : RegionName) (N : Nat) :
    attributedFull X Y N = attributedFullReference X Y N := rfl

/-- error: tuple assignment needs unsupported simultaneous register capture -/
#guard_msgs in
def broadcastOverwritesInput (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.fp32),("Y",.fp32)]} : CallSignature) in triton {
    a = tl.load(X)
    b = tl.load(Y)
    a, b = tl.broadcast(b, a)
    tl.store(Y, b)
  }

#axiomsClean attribute_broadcast_loop_preserves_casts
#axiomsClean broadcast_retains_each_precision
#axiomsClean attributed_constructor_keeps_output_conversion


/-- Ordinary numeric assignment creates an FP32 tensor. Both scalar carries
remain FP32 through an empty or nonempty source loop; no store cast is added. -/
def scalarLoopSignature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  constexpr := ["K"]

def scalarLoop (X Y : RegionName) (K : Nat) : ComputeKernel :=
  with_triton_signature scalarLoopSignature in triton {
    total = 0.0
    maximum = -float('inf')
    for i in range(K):
      x = tl.load(X + i)
      maximum = tl.maximum(maximum, x)
      total = total + tl.exp(x - maximum)
    tl.store(Y, total)
  }

def scalarLoopReference (X Y : RegionName) (K : Nat) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float32
  total = 0.0
  maximum = -float("inf")
  for i in tl.static_range($(K)) {
    x = tl.load(X + i)
    maximum = tl.maximum(maximum, x)
    total = total + tl.exp(x - maximum)
  }
  tl.store(Y, total)
}

theorem scalar_loop_dtype_and_store (X Y : RegionName) (K : Nat) :
    scalarLoop X Y K = scalarLoopReference X Y K := rfl

/-- An assigned float is a tensor; an inline literal is a weak operand.
The latter's FP16 result still needs a genuine FP32 output conversion. -/
def literalAssignment (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.fp16),("Y",.fp32)]} : CallSignature) in triton {
    zero = 0.0
    x = tl.load(X)
    assigned = zero + x
    tl.store(Y, assigned)
    inline = 0.0 + x
    tl.store(Y + 1, inline)
  }

def literalAssignmentReference (X Y : RegionName) : ComputeKernel := triton {
  tl.ports inputs=[X], outputs=[Y]
  tl.region X = tl.float16, Y = tl.float32
  zero = 0.0
  x = tl.load(X)
  assigned = zero + x
  tl.store(Y, assigned)
  inline = 0.0 + x
  tl.store(Y + 1, inline.to(tl.float32))
}

theorem assigned_and_inline_literal_precision (X Y : RegionName) :
    literalAssignment X Y = literalAssignmentReference X Y := rfl

/-- Tuple assignment materializes each numeric leaf with its own source type. -/
def literalTuple (Y : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("Y",.fp32)]} : CallSignature) in triton {
    a, b = 0.0, -float('inf')
    tl.store(Y, a)
    tl.store(Y + 1, b)
  }

theorem literal_tuple_no_invented_store_cast (Y : RegionName) :
    (literalTuple Y).surfaceBody.drop 1 =
      [.store .real [] (.region Y (.constNat 0)) (.alg (.ref .real [] "a")) .none,
       .store .real [] (.region Y (.constNat 1))
         (.alg (.ref .real [] "b")) .none] := rfl

/- Unsupported FP64 scalar materialization cannot silently enter an FP32
loop or erase a genuine output conversion. -/
/-- error: fp64 scalar literal assignment is not supported by this source frontend -/
#guard_msgs in
def fp64LiteralAssignment (Y : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("Y",.fp32)]} : CallSignature) in triton {
    x = 1e-50
    tl.store(Y, x)
  }

#axiomsClean scalar_loop_dtype_and_store
#axiomsClean assigned_and_inline_literal_precision
#axiomsClean literal_tuple_no_invented_store_cast

end VeriTile.Tests.TypedStores
