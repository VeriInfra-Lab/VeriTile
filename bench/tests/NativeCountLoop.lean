import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.NativeCountLoop
open VeriTile.Triton FP.Structural

def conversions (n : Nat) (z : Int) : ComputeKernel :=
  with_triton_call
    ({scalars := [("n",.integer),("z",.integer)]} : CallSignature)
    call_args [n := n, z := z] in
  triton {
    nf = n.to(tl.float32)
    zf = z.to(tl.float64)
  }

/-- Explicit integer conversions keep their destination precision and their
fromNat/fromInt channel. They are not floating casts or Real literals. -/
theorem conversion_channels (n : Nat) (z : Int) : conversions n z =
    .mk [] []
      [.assign .real [] "nf" (.compute (.alg .fp32 (.natToReal (.constNat n)))),
       .assign .real [] "zf" (.compute (.alg .fp64 (.intToReal (.constInt z))))] := rfl

/-- The actual frontend output executes those two distinct model operations. -/
theorem conversion_execution {α : Type} [Inhabited α] (M : Algebra α)
    (n : Nat) (z : Int) (s : State α) :
    exec M (conversions n z) s = some
      ((s.setReg "nf" .real [] (fun _ => M.fromNat (some .fp32) n)).setReg
        "zf" .real [] (fun _ => M.fromInt (some .fp64) z)) := by
  simp [conversion_channels, exec, run, step, evalExpr, evalComputeOp, evalOp_unfold]

def countLoop (N : Nat) : ComputeKernel :=
  with_triton_signature ({constexpr := ["N"]} : CallSignature) in
  triton {
    for i in range(N):
        value = i.to(tl.float32)
  }

/-- Specializing the immutable bound preserves every loop conversion. -/
theorem loop_channel (N : Nat) : countLoop N =
    .mk [] [] [.forLoop "i" N
      [.assign .real [] "value" (.compute (.alg .fp32 (.natToReal (.ref .nat [] "i"))))]] := rfl

/-- error: integer-to-float cast requires an explicit FP32 or FP64 destination -/
#guard_msgs in
def unsupportedHalf (n : Nat) : ComputeKernel :=
  with_triton_call ({scalars := [("n",.integer)]} : CallSignature)
    call_args [n := n] in
  triton {
    value = n.to(tl.float16)
  }

#axiomsClean conversion_channels
#axiomsClean conversion_execution
#axiomsClean loop_channel
end VeriTile.Tests.NativeCountLoop
