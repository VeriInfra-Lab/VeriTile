import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.SignedCast
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.CallArgumentCapture
open VeriTile.Triton FP.Structural

def signature : CallSignature where
  pointers := [("output",.fp32)]
  scalars := [("n",.i64)]

def source (output : RegionName) (value : Int) : ComputeKernel :=
  with_triton_call signature call_args [n := value] in triton {
    before = n.to(tl.float32)
    n = n.to(tl.int32)
    after = n.to(tl.float32)
    tl.store(output, before)
    tl.store(output + 1, after)
  }

/-- Earlier reads see the supplied value; later reads see the actual signed
cast. Replacing every occurrence by the call constant cannot satisfy this for
the valid i64 input 2^32+7 and arbitrary floating conversion primitives. -/
theorem captures_initial_value {α : Type} [Inhabited α] (M : Algebra α)
    (output : RegionName) (value : Int) (s : State α) :
    ∃ final, exec M (source output value) s = some final ∧
      final.mem output 0 = Cell.mk .real (M.fromInt (some .fp32) value) ∧
      final.mem output 1 = Cell.mk .real
        (M.fromInt (some .fp32) (BitVec.ofInt 32 value).toInt) := by
  simp [source,ComputeKernel.withScalarArguments,ScalarArgument.statement,
    ScalarArgument.integer,exec,ComputeKernel.surfaceBody,run,step,evalExpr,
    evalComputeOp,FP.Structural.eval_intToSigned,evalOp_unfold,ComputeDType.eraseDType,
    bop,numeric,store,State.setReg,State.write,TileShape.allIndices,Region.cast,
    ofFloat,toFloat,natLt,integralMod,IntegerExpr.sameShape,
    IntegerExpr.same_left,IntegerExpr.same_right]
  simpa [IntegerExpr.signedBits] using
    congrArg (M.fromInt (some .fp32)) (signedBits_int_mod 32 value)

/-- error: per-call specialization does not yet support reassigned scalar arguments -/
#guard_msgs in
example (output : RegionName) (value : Int) : ComputeKernel :=
  with_triton_call signature call_args [n := value] in triton {
    n = n + 1
    tl.store(output, n.to(tl.float32))
  }

#axiomsClean captures_initial_value

def loopSource (output : RegionName) (value : Int) : ComputeKernel :=
  with_triton_call signature call_args [n := value] in triton {
    n = tl.cast(n, tl.int64)
    base_offsets = tl.cast(tl.arange(0, 1), tl.int64)
    for i in range(0, n, 1):
        offsets = i + base_offsets
        mask = offsets < n
        tl.store(output + offsets, 1.0, mask=mask)
  }

/-- Signed induction variables and comparisons remain signed/Bool through
the pointer and mask inference. The positive range executes its active lane. -/
theorem masked_loop_one {α : Type} [Inhabited α] (M : Algebra α)
    (output : RegionName) (s : State α) :
    ∃ final, exec M (loopSource output 1) s = some final ∧
      final.mem output 0 = Cell.mk .real
        (M.cast (some .fp32) .real .real (M.literal (some .fp32) .real 1.0)) := by
  simp [loopSource,ComputeKernel.withScalarArguments,ScalarArgument.statement,
    ScalarArgument.integer,exec,ComputeKernel.surfaceBody,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,ComputeDType.eraseDType,
    bop,numeric,numericLt,natLt,store,State.setReg,State.write,
    TileShape.allIndices,List.finRange_succ,Region.cast]
  rw [rangeInt]
  simp [run,step,evalExpr,evalComputeOp,evalOp_unfold,ComputeDType.eraseDType,
    bop,numeric,numericLt,natLt,store,State.setReg,State.write,
    TileShape.allIndices,List.finRange_succ,Region.cast,ofFloat,toFloat]
  rw [rangeInt]
  simp

/-- A negative signed bound yields an empty range, with the entire memory
preserved. Clipping the bound into an unsigned value is unnecessary. -/
theorem masked_loop_negative {α : Type} [Inhabited α] (M : Algebra α)
    (output : RegionName) (s : State α) :
    ∃ final, exec M (loopSource output (-1)) s = some final ∧ final.mem = s.mem := by
  simp [loopSource,ComputeKernel.withScalarArguments,ScalarArgument.statement,
    ScalarArgument.integer,exec,ComputeKernel.surfaceBody,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,ComputeDType.eraseDType,
    bop,numeric,numericLt,natLt,store,State.setReg,State.write,Region.cast]
  rw [rangeInt]
  simp

#axiomsClean masked_loop_one
#axiomsClean masked_loop_negative

def negativeStart (output : RegionName) (limit : Int) : ComputeKernel :=
  with_triton_call signature call_args [n := limit] in triton {
    n = n.to(tl.int64)
    base_offsets = tl.arange(0, 1).to(tl.int64)
    for i in range(-1, n, 1):
        offsets = i + base_offsets
        mask = offsets < n
        value = i.to(tl.float32)
        tl.store(output + offsets + 1, value, mask=mask)
  }

/-- The negative first induction value reaches conversion and address
calculation unchanged. Both final addresses are valid: 0 and 1. -/
theorem negative_induction {α : Type} [Inhabited α] (M : Algebra α)
    (output : RegionName) (s : State α) :
    ∃ final, exec M (negativeStart output 1) s = some final ∧
      final.mem output 0 = Cell.mk .real (M.fromInt (some .fp32) (-1)) ∧
      final.mem output 1 = Cell.mk .real (M.fromInt (some .fp32) 0) := by
  simp [negativeStart,ComputeKernel.withScalarArguments,ScalarArgument.statement,
    ScalarArgument.integer,exec,ComputeKernel.surfaceBody,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,ComputeDType.eraseDType,
    bop,numeric,numericLt,natLt,store,State.setReg,State.write,
    TileShape.allIndices,List.finRange_succ,Region.cast]
  rw [rangeInt]
  simp [run,step,evalExpr,evalComputeOp,evalOp_unfold,ComputeDType.eraseDType,
    bop,numeric,numericLt,natLt,store,State.setReg,State.write,
    TileShape.allIndices,List.finRange_succ,Region.cast,ofFloat,toFloat]
  rw [rangeInt]
  simp [run,step,evalExpr,evalComputeOp,evalOp_unfold,ComputeDType.eraseDType,
    bop,numeric,numericLt,natLt,store,State.setReg,State.write,
    TileShape.allIndices,List.finRange_succ,Region.cast,ofFloat,toFloat]
  rw [rangeInt]
  simp

#axiomsClean negative_induction
end VeriTile.Tests.CallArgumentCapture
