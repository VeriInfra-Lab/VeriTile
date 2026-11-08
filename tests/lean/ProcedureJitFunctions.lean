import VeriTile.Triton.DSL.ProcedureJitFunctions
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.Structural
namespace VeriTile.Tests.ProcedureJitFunctions
open VeriTile.Triton FP.Structural
noncomputable section

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("stride",.integer)]
  constexpr := ["N"]

def helper : ProcedureJitFunction :=
  triton_procedure signature (X,Y,stride,N) constexpr_args [N] {
    for i in range(0, N):
        value = tl.load(X + i * stride)
        tl.store(Y + i, value)
    return
  }

def instantiated (X Y : RegionName) (stride N : Nat) : ComputeKernel :=
  instantiate_triton_procedure helper procedure_args [X := X,Y := Y,stride := stride,N := N]

def called (X Y : RegionName) (stride N : Nat) : ComputeKernel :=
  with_triton_call signature call_args [stride := stride]
    jit_procedures [copy := helper] in triton {
      copy(X, Y, stride, N=N)
    }

/-- A complete effect call and the same independently instantiated procedure
execute the same loads, stores and loop, for every size and arithmetic model. -/
theorem call_eq_instance (X Y : RegionName) (stride N : Nat) :
    called X Y stride N=instantiated X Y stride N := rfl

example (X Y : RegionName) (stride N : Nat) : True := by
  fail_if_success
    let bad : ComputeKernel :=
      with_triton_call signature call_args [stride := stride]
        jit_procedures [copy := helper] in triton {
          value = 0
          copy(X, Y, stride, N=N)
        }
  trivial

example (X Y : RegionName) (stride N : Nat) : True := by
  fail_if_success
    let bad : ComputeKernel :=
      with_triton_call signature call_args [stride := stride]
        jit_procedures [copy := helper] in triton {
          copy(X, Y, stride, N=N)
          tl.store(Y, value)
        }
  trivial


theorem one_store {α : Type} [Inhabited α] (M : Algebra α) (X Y : RegionName)
    (stride : Nat) (s : State α) :
    ∃ t, exec M (called X Y stride 1) s=some t ∧
      t.mem=(s.write Y 0 (.mk .real (M.fp32Load ((s.mem X 0).read .real)))).mem := by
  simp [called,exec,ComputeKernel.surfaceBody,run,step,range,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,numeric,bop,State.setReg,store,ComputeDType.eraseDType,ofFloat,toFloat,State.write]


def castHelper : ProcedureJitFunction :=
  triton_procedure signature (X,Y,stride,N) constexpr_args [N] {
    first = tl.load(X)
    second = tl.load(Y).to(first.dtype)
    tl.store(Y, second)
    return
  }
def castCalled (X Y : RegionName) (stride N : Nat) : ComputeKernel :=
  with_triton_call signature call_args [stride := stride] jit_procedures [copy := castHelper] in triton {
    copy(X, Y, stride, N=N)
  }
theorem explicit_load_cast {α : Type} [Inhabited α] (M : Algebra α) (X Y : RegionName)
    (stride N : Nat) (s : State α) :
    ∃ t, exec M (castCalled X Y stride N) s=some t ∧
      t.mem=(s.write Y 0 (.mk .real (M.cast (some .fp32) .real .real
        (M.fp32Load ((s.mem Y 0).read .real))))).mem := by
  simp [castCalled,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,numeric,bop,State.setReg,store,ComputeDType.eraseDType,ofFloat,toFloat,State.write]

/-- A caller must not override a callee's storage dtype. -/
theorem reject_mismatched_storage (X Y : RegionName) (stride N : Nat) : True := by
  fail_if_success
    let bad : ComputeKernel :=
      with_triton_call ({signature with pointers := [("X",.fp16),("Y",.fp32)]} : CallSignature)
        call_args [stride := stride] jit_procedures [copy := helper] in triton {
          copy(X, Y, stride, N=N)
        }
  trivial
#axiomsClean reject_mismatched_storage

#axiomsClean one_store
#axiomsClean explicit_load_cast

#axiomsClean call_eq_instance
end
end VeriTile.Tests.ProcedureJitFunctions
