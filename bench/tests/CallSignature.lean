import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Erasure
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
end VeriTile.Tests.CallSignature
