import VeriTile.Triton.DSL.StatementJitCallConstants
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.SignedCast
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.StatementJitCallConstants
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

def wrapping : StatementJitFunction :=
  triton_function (x, BIAS, OFFSET) constexpr_args [] {
    result = x - BIAS + OFFSET
    return result
  }
def signature : CallSignature where
  pointers := [("X",.u8)]
  constexpr := ["BIAS", "OFFSET"]
  readPorts := ["X"]
def wrapped (X : RegionName) : ComputeKernel :=
  let BIAS : Nat := 1
  let OFFSET : Nat := 127
  with_triton_call signature call_args [] jit_functions [wrapping := wrapping] in triton {
    x = tl.load(X)
    result = wrapping(x, BIAS, OFFSET)
  }

/-- The uint8 subtraction wraps before the addition, including x=0. No
floating backend operation can change either integer operation. -/
theorem two_byte_operations {α : Type} [Inhabited α] (M : Algebra α)
    (X : RegionName) (s : State α) :
    ∃ t, exec M (wrapped X) s=some t ∧
      t.regs .nat [] "result"=some (fun _=>
        (((((s.mem X 0).read .nat : Int)-1)%256).toNat+127)%256) := by
  cases X
  simp [wrapped,exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,State.setReg,ComputeDType.eraseDType,ofFloat,toFloat,
    numeric,bop,Broadcast.leftIndex,Broadcast.rightIndex,Region.cast]
  funext i
  simp only [integralMod]
  rw [max_eq_left (Int.emod_nonneg _ (by decide : (256 : Int) ≠ 0))]
  omega


def signedSelection : StatementJitFunction :=
  triton_function (x, EXP, MANTISSA) constexpr_args [] {
    shifted = x.to(tl.int32) << (EXP - MANTISSA)
    selected = tl.where(x == 0, 0, shifted)
    output = selected.to(tl.float32, bitcast=True)
    return output
  }
def selectSignature : CallSignature where
  pointers := [("X",.u8)]
  constexpr := ["EXP", "MANTISSA"]
  readPorts := ["X"]
def selected (X : RegionName) : ComputeKernel :=
  let EXP : Nat := 5
  let MANTISSA : Nat := 1
  with_triton_call selectSignature call_args []
    jit_functions [signedSelection := signedSelection] in triton {
    x = tl.load(X)
    result = signedSelection(x, EXP, MANTISSA)
  }

/-- The nonnegative where literal is promoted to signed32 and the final
conversion calls the backend's FP32 bit decoder, not its numerical cast. -/
theorem selection_bitcast {α : Type} [Inhabited α] (M : Algebra α)
    (X : RegionName) (s : State α) (zero : (s.mem X 0).read .nat=0) :
    ∃ t, exec M (selected X) s=some t ∧
      t.regs .real [] "result"=some (fun _=> M.fp32Bits ⟨BitVec.ofNat 32 0⟩) := by
  cases X
  simp [selected,exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,State.setReg,ComputeDType.eraseDType,ofFloat,toFloat,
    FP.Structural.eval_intToSigned, FP.Structural.eval_intToUnsigned, FP.Structural.eval_unsignedToInt,
    numeric,numericEq,numericLt,natLt,integralMod,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,Region.cast,zero,
    IntegerExpr.same_left,IntegerExpr.same_right]
  try rfl

/-- A nonzero byte takes the tensor branch and preserves its shifted bits. -/
theorem nonzero_bitcast {α : Type} [Inhabited α] (M : Algebra α)
    (X : RegionName) (s : State α) (byte : (s.mem X 0).read .nat=255) :
    ∃ t, exec M (selected X) s=some t ∧
      t.regs .real [] "result"=some (fun _=> M.fp32Bits ⟨BitVec.ofNat 32 4080⟩) := by
  cases X
  simp [selected,exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,State.setReg,ComputeDType.eraseDType,ofFloat,toFloat,
    FP.Structural.eval_intToSigned, FP.Structural.eval_intToUnsigned, FP.Structural.eval_unsignedToInt,
    numeric,numericEq,numericLt,natLt,integralMod,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,Region.cast,byte,
    IntegerExpr.same_left,IntegerExpr.same_right]
  try rfl

def shapeSignature : CallSignature where
  constexpr := ["N"]
def shapeSource (N : Nat) : ComputeKernel :=
  with_triton_call shapeSignature call_args [] jit_functions [] in triton {
    DOUBLE: tl.constexpr = N * 2
    offsets = tl.arange(0, DOUBLE).to(tl.int64)
  }
/-- Caller constexpr aliases preserve the complete output tile, also at N=0. -/
theorem output_shape {α : Type} [Inhabited α] (M : Algebra α) (N : Nat) (s : State α) :
    ∃ t, exec M (shapeSource N) s=some t ∧
      t.regs .int [N*2] "offsets"=some (fun i=>(i.1.val : Int)) := by
  simp [shapeSource,exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,State.setReg,ComputeDType.eraseDType,ofFloat,toFloat]
  try rfl

def hygieneSignature : CallSignature := {signature with scalars := [("value",.integer)]}
def hygiene (X : RegionName) : ComputeKernel :=
  let BIAS : Nat := 1
  let OFFSET : Nat := 127
  let __veritile_specialized_jit_0 : Int := 91
  with_triton_call hygieneSignature call_args [value := __veritile_specialized_jit_0]
    jit_functions [wrapping := wrapping] in triton {
    copied = value
    x = tl.load(X)
    result = wrapping(x, BIAS, OFFSET)
  }
/-- Metadata temporaries cannot capture a caller's Lean argument binding. -/
theorem metadata_hygiene {α : Type} [Inhabited α] (M : Algebra α)
    (X : RegionName) (s : State α) :
    ∃ t, exec M (hygiene X) s=some t ∧ t.regs .int [] "copied"=some (fun _=>91) := by
  simp [hygiene,exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,State.setReg,ComputeDType.eraseDType,ofFloat,toFloat,
    numeric,bop,Broadcast.leftIndex,Broadcast.rightIndex,Region.cast]

-- Type inference reads the outer pointer and the state at the call point.
run_cmd Lean.Elab.Command.liftTermElabM do
  let nested ← `(tritonExpr| tl.load(Y + tl.load(X)))
  if (DSL.JitIntegerPromotion.kind? [] [Lean.Name.mkSimple "X"] nested).isSome then
    throwError "nested byte offsets must not change float pointer storage"
  let body ← `(triton { X = Y x = tl.load(X) result = wrapping(x, 1, 127) })
  let `(triton { $stmts:tritonStmt* }) := body | throwError "body parse"
  let calls := DSL.JitIntegerPromotion.byteParameters [Lean.Name.mkSimple "X"]
    (Lean.Name.mkSimple "wrapping") [Lean.Name.mkSimple "x", Lean.Name.mkSimple "BIAS", Lean.Name.mkSimple "OFFSET"] stmts
  unless calls==[[]] do throwError "a rebound pointer cannot provide an earlier byte-storage fact"

def byteSignature : CallSignature := {signature with constexpr := []}

def tooWideHelper : StatementJitFunction :=
  triton_function (x) constexpr_args [] { return x + 256 }
/-- error: uint8 weak scalar outside the supported range 0..255 -/
#guard_msgs in
def tooWide (X : RegionName) : ComputeKernel :=
  with_triton_call byteSignature call_args [] jit_functions [convert := tooWideHelper] in triton {
    x = tl.load(X)
    result = convert(x=x)
  }

def negativeHelper : StatementJitFunction :=
  triton_function (x) constexpr_args [] { return x + -1 }
/-- error: uint8 weak scalar outside the supported range 0..255 -/
#guard_msgs in
def negativeScalar (X : RegionName) : ComputeKernel :=
  with_triton_call byteSignature call_args [] jit_functions [convert := negativeHelper] in triton {
    x = tl.load(X)
    result = convert(x=x)
  }

def signedOverflowHelper : StatementJitFunction :=
  triton_function (x) constexpr_args [] {
    y = x.to(tl.int32)
    return tl.where(x == 0, 2147483648, y)
  }
/-- error: weak integer where branch does not fit the signed32 tensor -/
#guard_msgs in
def signedOverflow (X : RegionName) : ComputeKernel :=
  with_triton_call byteSignature call_args [] jit_functions [convert := signedOverflowHelper] in triton {
    x = tl.load(X)
    result = convert(x=x)
  }

def leftShiftHelper : StatementJitFunction :=
  triton_function (x) constexpr_args [] {
    result = x << 4
    return result
  }
/-- error: uint8 left shifts require an explicit source-width lowering -/
#guard_msgs in
def unsupportedByteShift (X : RegionName) : ComputeKernel :=
  with_triton_call byteSignature call_args [] jit_functions [convert := leftShiftHelper] in triton {
    x = tl.load(X)
    result = convert(x=x)
  }

def runtimeHelper : StatementJitFunction :=
  triton_function (x, y) constexpr_args [] { return x - y }
def runtimeSignature : CallSignature := {byteSignature with scalars := [("y",.integer)]}
/-- error: uint8 arithmetic requires another byte tensor or a closed weak integer -/
#guard_msgs in
def unsupportedMixedWidth (X : RegionName) (y : Int) : ComputeKernel :=
  with_triton_call runtimeSignature call_args [y := y] jit_functions [convert := runtimeHelper] in triton {
    x = tl.load(X)
    result = convert(x=x, y=y)
  }

#axiomsClean output_shape
#axiomsClean metadata_hygiene
#axiomsClean two_byte_operations
#axiomsClean selection_bitcast
#axiomsClean nonzero_bitcast
end
end VeriTile.Tests.StatementJitCallConstants
