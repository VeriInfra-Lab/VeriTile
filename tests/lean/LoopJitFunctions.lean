import VeriTile.Triton.DSL.LoopJitFunctions
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.LoopJitFunctions
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2500000
set_option maxRecDepth 24000
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false

def helper : StatementJitFunction := triton_function (a,b,N,BLOCK) constexpr_args [N,BLOCK] {
  lo, hi = 0, N
  for i in range(lo, hi, BLOCK):
    a = a + b
    b = a * b
  return b, a
}
def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("N",.integer)]
  constexpr := ["BLOCK"]
def source (X Y : RegionName) (N BLOCK : Nat) : ComputeKernel :=
  with_triton_call signature call_args [N := N] jit_functions [f := helper] in triton {
    a = tl.load(X)
    b = tl.load(Y)
    a, b = f(a, b, N, BLOCK)
    tl.store(X, a)
    tl.store(Y, b)
  }
def initial {A : Type} [Inhabited A] (x y : A) : State A :=
  {mem := fun r _=>.mk .real (if r="X" then x else y),regs := fun _ _ _=>none,
   pids := fun _=>0,numPids := fun _=>0,undef := fun dtype _ _=>defaultValue dtype}
def expected {A : Type} (M : Algebra A) (x y : A) : A × A :=
  let x:=M.fp32Load x
  let y:=M.fp32Load y
  let a1:=M.binary (some .fp32) .real .add x y
  let b1:=M.binary (some .fp32) .real .mul a1 y
  let a2:=M.binary (some .fp32) .real .add a1 b1
  let b2:=M.binary (some .fp32) .real .mul a2 b1
  (b2,a2)
/-- A runtime bound, both loop-carried values and reversed simultaneous
returns execute the original rounded operation tree on the loaded operands. -/
theorem complete_execution {A : Type} [Inhabited A] (M : Algebra A) (x y : A) :
    ∃ t, exec M (source "X" "Y" 2 1) (initial x y)=some t ∧
      t.mem "X" 0=.mk .real (expected M x y).1 ∧
      t.mem "Y" 0=.mk .real (expected M x y).2 := by
  simp [source,initial,expected,exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,State.setReg,State.write,store,
    ofFloat,toFloat,numeric,numericLt,natLt,bop,Region.cast,ComputeDType.eraseDType]

  simp [run,step,range,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,State.setReg,State.write,store,
    ofFloat,toFloat,numeric,numericLt,natLt,bop,Region.cast,ComputeDType.eraseDType]

def halfSignature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp16)]
  scalars := [("N",.integer)]
  constexpr := ["BLOCK"]
def halfSource (X Y : RegionName) (N BLOCK : Nat) : ComputeKernel :=
  with_triton_call halfSignature call_args [N := N] jit_functions [f := helper] in triton {
    a = tl.load(X)
    b = tl.load(X)
    a, b = f(a, b, N, BLOCK)
    tl.store(Y, a.to(Y.type.element_ty))
  }
def halfExpected (X Y : RegionName) (N BLOCK : Nat) : ComputeKernel :=
  with_triton_call halfSignature call_args [N := N] jit_functions [f := helper] in triton {
    a = tl.load(X)
    b = tl.load(X)
    a, b = f(a, b, N, BLOCK)
    tl.store(Y, a.to(tl.float16))
  }
/-- Resolve caller pointer metadata while retaining the explicit FP32-to-half
conversion and the complete helper's loop, captures and tuple returns. -/
theorem half_output_cast (X Y : RegionName) (N BLOCK : Nat) :
    halfSource X Y N BLOCK=halfExpected X Y N BLOCK := rfl

def collisionSource (X Y : RegionName) (N BLOCK : Nat) : ComputeKernel :=
  with_triton_call signature call_args [N := N] jit_functions [f := helper] in triton {
    __veritile_statement_jit_0 = tl.load(X)
    a = tl.load(X)
    b = tl.load(Y)
    a, b = f(a, b, N, BLOCK)
  }
/-- Helper capture names cannot overwrite an identically named caller local.
An empty loop also returns the original values in simultaneous swapped order. -/
theorem caller_local_preserved {A : Type} [Inhabited A] (M : Algebra A) (x y : A) :
    ∃ t, exec M (collisionSource "X" "Y" 0 1) (initial x y)=some t ∧
      t.regs .real [] "__veritile_statement_jit_0"=some (fun _=>M.fp32Load x) ∧
      t.regs .real [] "a"=some (fun _=>M.fp32Load y) ∧
      t.regs .real [] "b"=some (fun _=>M.fp32Load x) := by
  simp [collisionSource,initial,exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,State.setReg,ofFloat,toFloat,
    numeric,numericLt,natLt,bop,Region.cast,ComputeDType.eraseDType]
  simp [range,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,State.setReg,
    ofFloat,toFloat,numeric,numericLt,natLt,bop,Region.cast,ComputeDType.eraseDType]

example : True := by
  fail_if_success
    let bad : StatementJitFunction := triton_function (a,N,BLOCK) constexpr_args [N,BLOCK] {
      for N in range(0, N, BLOCK):
        a += a
      return a
    }
  trivial
example : True := by
  fail_if_success
    let bad : StatementJitFunction := triton_function (a,N,BLOCK) constexpr_args [N,BLOCK] {
      for i in range(0, N, BLOCK):
        a = a + future
      future = a
      return a
    }
  trivial
example (X Y : RegionName) (N BLOCK : Nat) : True := by
  fail_if_success
    let bad : ComputeKernel := with_triton_call halfSignature call_args [N := N]
        jit_functions [f := helper] in triton {
      Y = X
      a = tl.load(X)
      b = tl.load(X)
      a, b = f(a, b, N, BLOCK)
      tl.store(Y, a.to(Y.type.element_ty))
    }
  trivial
example (X Y : RegionName) (N BLOCK : Nat) : True := by
  fail_if_success
    let bad : ComputeKernel := with_triton_call signature call_args [N := N]
        jit_functions [f := helper] in triton {
      a = tl.load(X)
      b = tl.load(Y)
      a = f(a, b, N, BLOCK)
      tl.store(Y, a)
    }
  trivial

#axiomsClean complete_execution
#axiomsClean half_output_cast
#axiomsClean caller_local_preserved
end
end VeriTile.Tests.LoopJitFunctions
