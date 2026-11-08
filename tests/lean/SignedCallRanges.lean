import VeriTile.Triton.DSL.SignedCallRanges
import VeriTile.Triton.SignedSource
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.SignedCallRanges
open VeriTile.Triton
noncomputable section
def signature : CallSignature where
  pointers := [("Y",.fp32)]
  scalars := [("N",.integer)]
  readPorts := []
  writePorts := ["Y"]
def pasted (Y : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_call signature call_args [N := N] in triton {
    N = tl.cast(N, tl.int64)
    for k in range(0, N):
        tl.store(Y + k, 0.0)
  }
/-- Literal-zero two-argument Python ranges retain an Int64 counter when the
bound has a dominating Int64 self-cast. -/
theorem signed_counter (Y : RegionName) (N : Nat) :
    (Source.signedRange? (pasted Y N).surfaceBody "k").isSome=Bool.true := rfl
/-- The empty signed source loop succeeds without any floating conversion
callback and does not touch memory. -/
theorem empty_loop {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (Y : RegionName) (s : FP.Structural.State α) :
    ∃ t,FP.Structural.exec M (pasted Y 0) s=some t ∧ t.mem=s.mem := by
  refine ⟨s.setReg "N" .int [] (fun _ => 0), ?_, rfl⟩
  simp only [FP.Structural.exec,pasted,FP.Structural.run,FP.Structural.step,
    FP.Structural.evalExpr,FP.Structural.evalOp_unfold,bind,pure,Option.bind]
  rw [FP.Structural.rangeInt]
  simp

#axiomsClean signed_counter
#axiomsClean empty_loop
end
end VeriTile.Tests.SignedCallRanges
