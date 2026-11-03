import VeriTile.Triton.DSL.MutableScalarCalls
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.MutableScalarCalls
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false

def signature : CallSignature where
  pointers := [("Lengths",.i64),("Y",.i64)]
  scalars := [("n",.integer),("stride",.integer)]
  constexpr := ["VARLEN"]
def source (Lengths Y : RegionName) (n : Int) (stride : Nat) (VARLEN : Bool) : ComputeKernel :=
  with_triton_call signature call_args [n := n, stride := stride] in triton {
    seen = n
    if VARLEN:
      n = tl.load(Lengths)
    n = n + stride
    tl.store(Y, n)
  }
/-- Initial argument binding precedes the copied body. Runtime assignment sees
actual metadata, while reads preceding the assignment retain the original
argument. Signed inputs, both constexpr paths and arbitrary FP models remain. -/
theorem initial_and_reassigned_values {A : Type} [Inhabited A] (M : Algebra A)
    (s : State A) (n m : Int) (stride : Nat) (VARLEN : Bool)
    (hm : (s.mem "Lengths" 0).read .int=m) :
    ∃ t, exec M (source "Lengths" "Y" n stride VARLEN) s=some t ∧
      t.regs .int [] "seen"=some (fun _=>n) ∧
      t.mem "Y" 0=.mk .int ((if VARLEN then m else n)+(stride : Int)) := by
  cases VARLEN <;>
    simp [source,exec,ComputeKernel.withScalarArguments,ScalarArgument.statement,ScalarArgument.integer,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,hm,Option.bind,State.setReg,State.write,store,numeric,bop,
      Region.cast,ofFloat,toFloat,ComputeDType.eraseDType]

example : True := by
  fail_if_success
    let bad : ComputeKernel :=
      let VARLEN := Bool.true
      with_triton_call signature call_args [n := (2 : Nat), secret := (1 : Nat)] in triton {
        n = tl.load(Lengths)
      }
  trivial
example : True := by
  fail_if_success
    let bad : ComputeKernel :=
      let VARLEN := Bool.true
      with_triton_call signature call_args [n := (2 : Nat)] in triton {
        n = tl.load(Lengths)
      }
  trivial
example : True := by
  fail_if_success
    let bad : ComputeKernel :=
      let VARLEN := Bool.true
      with_triton_call signature call_args [n := (2 : Nat), stride := (1 : Nat)] in triton {
        VARLEN = False
        n = tl.load(Lengths)
      }
  trivial

example : True := by
  fail_if_success
    let bad : ComputeKernel :=
      let VARLEN := Bool.true
      with_triton_call {signature with pointers := [("stride",.i64),("Lengths",.i64),("Y",.i64)]}
        call_args [n := (2 : Nat), stride := (1 : Nat)] in triton {
        n = tl.load(Lengths)
        tl.store(Y, n)
      }
  trivial

#axiomsClean initial_and_reassigned_values
end
end VeriTile.Tests.MutableScalarCalls
