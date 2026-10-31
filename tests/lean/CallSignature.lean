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

def copySignature : Triton.CallSignature where
  pointers := [("metadata",.i32),("X",.fp32),("Y",.fp32)]

def copyAfterCast (metadata X Y : RegionName) : ComputeKernel :=
  with_triton_signature copySignature in triton {
    index = tl.cast(tl.load(metadata), tl.int64)
    value = tl.load(X + index)
    tl.store(Y + index, value)
  }

/-- A functional cast must read the declared integer cell, and storing an
already-FP32 load introduces no additional numerical callback. -/
theorem integer_load_and_identity_store (metadata X Y : RegionName) :
    copyAfterCast metadata X Y = (triton {
      tl.region metadata = tl.int32, X = tl.float32, Y = tl.float32
      index = tl.load(metadata).to(tl.int64)
      value = tl.load(X + index)
      tl.store(Y + index, value)
    }) := rfl

def byteSignature : Triton.CallSignature where
  pointers := [("X",.u8),("Y",.u8)]

def copyByte (X Y : RegionName) : ComputeKernel :=
  with_triton_signature byteSignature in triton {
    value = tl.load(X)
    tl.store(Y, value)
  }

/-- The Nat carrier does not certify an eight-bit range: its destination
conversion must remain even when both source and destination use that carrier. -/
theorem byte_store_keeps_narrowing (X Y : RegionName) :
    copyByte X Y = (triton {
      tl.region X = tl.uint8, Y = tl.uint8
      value = tl.load(X)
      tl.store(Y, value.to(tl.uint8))
    }) := rfl

def fp32CopySignature : Triton.CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]

def inlineCopy (X Y : RegionName) : ComputeKernel :=
  with_triton_signature fp32CopySignature in triton {
    tl.store(Y, tl.load(X))
  }

/-- A nested load has the same FP32 storage interpretation as an assigned
load; the source spelling must not erase its input rounding boundary. -/
theorem nested_load_keeps_storage {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (X Y : RegionName) (s : FP.Structural.State α) :
    ∃ t, FP.Structural.exec M (inlineCopy X Y) s = some t ∧
      t.mem Y 0 = FP.Structural.Cell.mk .real (M.fp32Load ((s.mem X 0).read .real)) := by
  simp [inlineCopy,FP.Structural.exec,FP.Structural.run,FP.Structural.step,
    FP.Structural.evalExpr,FP.Structural.evalComputeOp,FP.Structural.evalOp_unfold,
    FP.Structural.store,FP.Structural.evalOffsets,ComputeKernel.surfaceBody,
    ComputeDType.eraseDType,Region.cast,Option.bind]

def signedIndexSignature : Triton.CallSignature where
  pointers := [("Y",.fp32)]
  constexpr := ["BASE"]

def signedIndex (Y : RegionName) (BASE : Nat) : ComputeKernel :=
  with_triton_signature signedIndexSignature in triton {
    linear = tl.cast(tl.program_id(0), tl.int64) + BASE
    tl.store(Y + linear, tl.full([], 0.0, dtype=tl.float32))
  }

/-- Repairing functional load casts must not change signed inference for
an enclosing arithmetic expression such as a grid-base calculation. -/
theorem signed_cast_in_arithmetic (Y : RegionName) (BASE : Nat) :
    signedIndex Y BASE = (triton {
      tl.region Y = tl.float32
      linear = tl.cast(tl.program_id(0), tl.int64) + BASE
      tl.store(Y + linear, tl.full([], 0.0, dtype=tl.float32))
    }) := rfl

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
end VeriTile.Tests.CallSignature
