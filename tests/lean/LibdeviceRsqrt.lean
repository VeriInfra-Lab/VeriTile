import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.LibdeviceRsqrt
open VeriTile.Triton

def direct : ComputeKernel := triton {
  x = (1.0).to(tl.float32)
  y = libdevice.rsqrt(x)
}
def qualified : ComputeKernel := triton {
  x = (1.0).to(tl.float32)
  y = tl.extra.cuda.libdevice.rsqrt(x)
}
def intrinsic : ComputeKernel := triton {
  x = (1.0).to(tl.float32)
  y = tl.rsqrt(x)
}
example : direct = qualified := rfl
example : direct.surfaceBody ≠ intrinsic.surfaceBody := by
  intro h
  simp only [direct, intrinsic, ComputeKernel.surfaceBody, List.cons.injEq] at h
  cases h.2.1

/-- Nested source loads must remain in the inferred input ports. -/
def memory (X Y : RegionName) : ComputeKernel := triton {
  tl.region X = tl.float32
  tl.region Y = tl.float32
  value = libdevice.rsqrt(tl.load(X).to(tl.float32))
  tl.store(Y, value)
}
example (X Y : RegionName) : (memory X Y).inputs = [X] := rfl

/-- FP execution calls the distinct, unconstrained libdevice backend. -/
theorem backend {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (p : Option ComputeDType) (s : FP.Structural.State α) (a : Op .real shape) :
    FP.Structural.evalOp M p (.libdeviceRsqrt a) s =
      ((FP.Structural.evalOp M p a s).map (fun v => M.unary p .libdeviceRsqrt ∘ v)) := by
  simp only [FP.Structural.evalOp, Option.map_eq_bind]
  rfl

example (a : Op .real shape) :
    (Op.libdeviceRsqrt a).eraseDType = Op.libdeviceRsqrt a.eraseDType := by rw [Op.eraseDType.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Tests.LibdeviceRsqrt
