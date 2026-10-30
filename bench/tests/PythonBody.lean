import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.PythonBody
open VeriTile.Triton

/-- error: nested load source precision is unsupported by the current frontend -/
#guard_msgs in
def nestedLoad (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.fp32),("Y",.fp32)]} : CallSignature) in
  triton {
    tl.store(Y, tl.max(tl.load(X + tl.arange(0, 4)), axis=0))
  }

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("count",.i32),("stride",.i64)]
  constexpr := ["B"]

def pasted (X Y : RegionName) (B count : Nat) (stride : Int) : ComputeKernel :=
  with_triton_call signature call_args [count := count, stride := stride] in
  triton {
    """A Python docstring can contain } $ tl.region and
def fake_definition:
end fake_namespace
    without terminating the quotation."""
    # } $ tl.param and -- are ordinary Python comment text.
    offsets = tl.arange(0, B).to(tl.int64)
    active = offsets < count
    x = tl.load(
        X + offsets * stride,
        mask=active, other=0,
    ).to(tl.float32)
    if count > 0:
        y = x--x
        tl.store(Y + offsets, y, mask=active)
    else:
        tl.store(Y + offsets, x, mask=active)
  }

def expanded (X Y : RegionName) (B count : Nat) (stride : Int) : ComputeKernel :=
  triton {
    tl.ports inputs=[X], outputs=[Y]
    tl.region X = tl.float32, Y = tl.float32
    offsets = tl.arange(0, B).to(tl.int64)
    active = offsets < $((count : Nat))
    x = tl.load(X + offsets * $((stride : Int)), mask=active, other=0).to(tl.float32)
    if $((count : Nat)) > 0 {
      y = x - -x
      tl.store(Y + offsets, y, mask=active)
    } else {
      tl.store(Y + offsets, x, mask=active)
    }
  }

theorem original_comments_indentation_and_integer_calls
    (X Y : RegionName) (B count : Nat) (stride : Int) :
    pasted X Y B count stride = expanded X Y B count stride := rfl

def integerSignature : CallSignature where
  scalars := [("n",.i64)]

def scalarCast (n : Int) : ComputeKernel :=
  with_triton_call integerSignature call_args [n := n] in
  triton {
    value = n.to(tl.int64)
  }

/-- Signed values supplied by a call reach the signed register unchanged. -/
theorem signed_call_executes {A : Type} [Inhabited A] (M : FP.Structural.Algebra A)
    (s : FP.Structural.State A) (n : Int) :
    (FP.Structural.exec M (scalarCast n) s).bind
      (fun t => t.regs .int [] "value") = some (fun _ => n) := by
  simp [scalarCast, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
    FP.Structural.evalOp_unfold, FP.Structural.State.setReg]

theorem scalar_cast_is_retained (n : Int) : scalarCast n = triton {
    value = ($((n : Int))).to(tl.int64)
  } := rfl

/-- error: every integer runtime scalar needs an explicit call binding -/
#guard_msgs in
def missingBinding (n : Nat) : ComputeKernel :=
  with_triton_call signature call_args [count := n] in
  triton { value = count }

/-- error: duplicate call argument binding -/
#guard_msgs in
def duplicateBinding (n : Int) : ComputeKernel :=
  with_triton_call integerSignature call_args [n := n, n := n] in
  triton { value = n }

/-- error: per-call specialization does not yet support reassigned scalar arguments -/
#guard_msgs in
def assignedArgument (n : Int) : ComputeKernel :=
  with_triton_call integerSignature call_args [n := n] in
  triton {
    n += 1
  }

/-- error: per-call specialization does not yet support reassigned scalar arguments -/
#guard_msgs in
def shadowedArgument (n : Int) : ComputeKernel :=
  with_triton_call integerSignature call_args [n := n] in
  triton {
    for n in range(4):
        value = n
  }

/-- error: per-call specialization does not yet support reassigned scalar arguments -/
#guard_msgs in
def tupleArgument (n : Int) : ComputeKernel :=
  with_triton_call integerSignature call_args [n := n] in
  triton {
    n, other = 1, 2
  }

/-- error: observed memory access is missing from the declared call ports -/
#guard_msgs in
def incorrectMemoryRole (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.fp32),("Y",.fp32)], readPorts := ["Y"], writePorts := ["Y"]} : CallSignature) in
  triton {
    value = tl.load(X)
    tl.store(Y, value)
  }

/-- error: per-call specialization does not yet support reassigned scalar arguments -/
#guard_msgs in
def tupleCallArgument (n : Int) : ComputeKernel :=
  with_triton_call integerSignature call_args [n := n] in
  triton {
    n, other = tl.max(tl.arange(0, 4), axis=0, return_indices=True)
  }

/-- error: per-call specialization does not yet support reassigned scalar arguments -/
#guard_msgs in
def positionalReductionArgument (n : Int) : ComputeKernel :=
  with_triton_call integerSignature call_args [n := n] in
  triton {
    n = tl.max(tl.arange(0, 4), 0)
  }

/-- A final function return is retained in the pasted source and denotes the
same complete execution as the function's implicit end. -/
def terminalReturn (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.fp32),("Y",.fp32)]} : CallSignature) in
  triton {
    value = tl.load(X)
    tl.store(Y, value)
    return
  }
def implicitReturn (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.fp32),("Y",.fp32)]} : CallSignature) in
  triton {
    value = tl.load(X)
    tl.store(Y, value)
  }
theorem terminal_return_preserves_whole_program (X Y : RegionName) :
    terminalReturn X Y = implicitReturn X Y := rfl

def onlyReturn : ComputeKernel :=
  with_triton_signature ({} : CallSignature) in triton { return }
theorem return_only_is_empty : onlyReturn = ComputeKernel.mk [] [] [] := rfl

/-- error: `return` is only supported as the sole body statement of `if cond { return }` -/
#guard_msgs in
def loopReturn : ComputeKernel :=
  with_triton_signature ({} : CallSignature) in triton {
    for i in range(2):
        return
  }

/-- error: `return` is only supported as the sole body statement of `if cond { return }` -/
#guard_msgs in
def nonterminalReturn (X : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.fp32)]} : CallSignature) in triton {
    return
    tl.store(X, 0.0)
  }

#axiomsClean terminal_return_preserves_whole_program
#axiomsClean return_only_is_empty

#axiomsClean original_comments_indentation_and_integer_calls
#axiomsClean scalar_cast_is_retained
#axiomsClean signed_call_executes
end VeriTile.Tests.PythonBody

namespace VeriTile.Tests.PythonBody.ReservedAndControlFlow
open VeriTile.Triton FP.Structural
noncomputable section

def keyword (p : RegionName) : ComputeKernel :=
  let «do» := p
  with_triton_signature ({pointers := [("do",.fp32)]} : CallSignature) in
  triton {
    _ = tl.load(do).to(tl.float32)
    copied = _
  }

/-- The source spelling denotes real names, including reads of `_`. -/
theorem keyword_names (p : RegionName) : keyword p = (let P : RegionName := p; triton {
    tl.ports inputs=[P], outputs=[]
    tl.region P = tl.float32
    «_» = tl.load(P).to(tl.float32)
    copied = «_»
  }) := rfl


def augmented (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.fp16),("Y",.fp32)]} : CallSignature) in
  triton {
    acc = tl.zeros([4], dtype=tl.float32)
    acc += tl.load(X + tl.arange(0, 4)).to(tl.float32)
    tl.store(Y + tl.arange(0, 4), acc)
  }
/-- The FP16 load, explicit FP32 conversion and FP32 add all survive. -/
theorem augmented_precision (X Y : RegionName) : augmented X Y = triton {
    tl.ports inputs=[X], outputs=[Y]
    tl.region X = tl.float16, Y = tl.float32
    acc = tl.zeros([4], dtype=tl.float32)
    acc += tl.load(X + tl.arange(0, 4)).to(tl.float32)
    tl.store(Y + tl.arange(0, 4), acc)
  } := rfl

/-- error: nested load source precision is unsupported by the current frontend -/
#guard_msgs in
def nestedAugmented (X : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.fp16)]} : CallSignature) in
  triton {
    acc = tl.zeros([4], dtype=tl.float32)
    acc += 2.0 * tl.load(X + tl.arange(0, 4)).to(tl.float32)
  }

def branched (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.fp32),("Y",.fp32)]} : CallSignature) in
  triton {
    i = tl.program_id(0)
    if i > 1:
        value = tl.load(X).to(tl.float32)
    elif i > 0:
        value = tl.load(X + 1).to(tl.float32)
    else:
        value = tl.full([], 0, tl.float32)
    tl.store(Y, value)
  }
/-- All branches retain their load boundary and common dtype. No redundant
store conversion is introduced when the source spelled `elif`. -/
theorem branch_precision (X Y : RegionName) : branched X Y = triton {
    tl.ports inputs=[X], outputs=[Y]
    tl.region X = tl.float32, Y = tl.float32
    i = tl.program_id(0)
    if i > 1 {
      value = tl.load(X).to(tl.float32)
    } else {
      if i > 0 {
        value = tl.load(X + 1).to(tl.float32)
      } else {
        value = tl.full([], 0, tl.float32)
      }
    }
    tl.store(Y, value)
  } := rfl

#axiomsClean keyword_names
#axiomsClean augmented_precision
#axiomsClean branch_precision
end
end VeriTile.Tests.PythonBody.ReservedAndControlFlow



namespace VeriTile.Tests.PythonBody.ParenthesizedAssignment
open VeriTile.Triton

def signature : CallSignature where
  scalars := [("n",.integer)]

def pasted (n : Int) : ComputeKernel :=
  with_triton_call signature call_args [n := n] in triton {
    a, b, c = (tl.cast(n, tl.int64), tl.cast(n + 1, tl.int64), tl.cast(n - 1, tl.int64))
  }

def unparenthesized (n : Int) : ComputeKernel :=
  with_triton_call signature call_args [n := n] in triton {
    a, b, c = tl.cast(n, tl.int64), tl.cast(n + 1, tl.int64), tl.cast(n - 1, tl.int64)
  }

theorem same_complete_program (n : Int) : pasted n = unparenthesized n := rfl

/-- Every original cast/value reaches the corresponding register in order. -/
theorem independent_tuple_executes {A : Type} [Inhabited A]
    (M : FP.Structural.Algebra A) (s : FP.Structural.State A) (n : Int) :
    (FP.Structural.exec M (pasted n) s).bind
      (fun t => do
        let a ← t.regs .int [] "a"
        let b ← t.regs .int [] "b"
        let c ← t.regs .int [] "c"
        return (a PUnit.unit, b PUnit.unit, c PUnit.unit)) = some (n, n + 1, n - 1) := by
  simp [pasted, FP.Structural.exec, ComputeKernel.surfaceBody, FP.Structural.run,
    FP.Structural.step, FP.Structural.evalExpr, FP.Structural.evalOp_unfold,
    FP.Structural.State.setReg, FP.Structural.numeric, FP.Structural.bop,
    ComputeDType.eraseDType]

/-- error: tuple assignment needs unsupported simultaneous register capture -/
#guard_msgs in
def dependent_swap (n : Int) : ComputeKernel :=
  with_triton_call signature call_args [n := n] in triton {
    a = n
    b = n + 1
    a, b = (b, a)
  }

/-- error: tuple assignment needs one value per target -/
#guard_msgs in
def incorrect_arity (n : Int) : ComputeKernel :=
  with_triton_call signature call_args [n := n] in triton {
    a, b = (n, n + 1, n - 1)
  }

#axiomsClean same_complete_program
#axiomsClean independent_tuple_executes
end VeriTile.Tests.PythonBody.ParenthesizedAssignment
