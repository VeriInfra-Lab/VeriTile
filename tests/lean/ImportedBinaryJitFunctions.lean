import VeriTile.Triton.DSL.ImportedBinaryJitFunctions
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.ImportedBinaryJitFunctions
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 12000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def power : ImportedBinaryJitFunction :=
  with_triton_imports [_pow := "triton.language.extra.cuda.libdevice.pow"] in triton_function (x, exponent) {
    return _pow(x.to(tl.float32), exponent)
  }
def signature : CallSignature where
  scalars := [("a",.fp32),("b",.fp32)]

def source (a : ℝ) : ComputeKernel :=
  with_triton_call signature call_args [a := a]
    signed_pointer_offsets [] binary_functions [power := power] in triton {
    result = power(a, b)
  }

theorem actual_value {α : Type} [Inhabited α] (M : Algebra α) (a : ℝ) (b : Values α .real []) (s : State α)
    (hb : s.regs .real [] "b"=some b) :
    ∃ t, exec M (source a) s = some t ∧
      t.regs .real [] "result" = some (fun _ =>
        M.binary (some .fp32) .real .pow
          (M.cast (some .fp32) .real .real (M.literal (some .fp32) .real a))
          (b PUnit.unit)) := by
  simp [source,hb,ComputeKernel.surfaceBody,exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,State.setReg,ComputeDType.eraseDType,ofFloat,toFloat]
  rfl


/-- Captures cannot overwrite a source register or leak across calls. -/
def hygienicSource (a : ℝ) : ComputeKernel :=
  with_triton_call signature call_args [a := a]
    signed_pointer_offsets [] binary_functions [power := power] in triton {
    __veritile_binary_power_0 = tl.full((), 17, tl.int64)
    left = power(a, b)
    right = power(a, left)
  }
theorem repeated_calls {α : Type} [Inhabited α] (M : Algebra α) (a : ℝ)
    (b : Values α .real []) (s : State α) (hb : s.regs .real [] "b"=some b) :
    let value := M.cast (some .fp32) .real .real (M.literal (some .fp32) .real a)
    ∃ t, exec M (hygienicSource a) s = some t ∧
      t.regs .int [] "__veritile_binary_power_0"=some (fun _ => 17) ∧
      t.regs .real [] "right"=some (fun _ => M.binary (some .fp32) .real .pow value
        (M.binary (some .fp32) .real .pow value (b PUnit.unit))) := by
  simp [hygienicSource,hb,exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,State.setReg,ComputeDType.eraseDType,ofFloat,toFloat]
  rfl

/- Source/alias validation also applies to manually constructed metadata. -/
/-- error: power helper requires its complete copied cast-and-power return expression -/
#guard_msgs in
def missingCast : ImportedBinaryJitFunction :=
  with_triton_imports [_pow := "triton.language.extra.cuda.libdevice.pow"] in triton_function (x, exponent) {
    return _pow(x, exponent)
  }
/-- error: unsupported binary helper extern binding -/
#guard_msgs in
def wrongImport : ImportedBinaryJitFunction :=
  with_triton_imports [_pow := "triton.language.math.pow"] in triton_function (x, exponent) {
    return _pow(x.to(tl.float32), exponent)
  }
/-- error: binary JIT helpers require a complete plain assignment with scalar/tensor register arguments -/
#guard_msgs in
def escapes (a : ℝ) : ComputeKernel :=
  with_triton_call signature call_args [a := a]
    signed_pointer_offsets [] binary_functions [power := power] in triton {
    escaped = power
  }
/-- error: binary JIT helpers require a complete plain assignment with scalar/tensor register arguments -/
#guard_msgs in
def rebound (a : ℝ) : ComputeKernel :=
  with_triton_call signature call_args [a := a]
    signed_pointer_offsets [] binary_functions [power := power] in triton {
    result = power(a, b)
    power = 0
  }
/-- error: duplicate or captured binary helper name -/
#guard_msgs in
def capturesArgument (a : ℝ) : ComputeKernel :=
  with_triton_call signature call_args [a := a]
    signed_pointer_offsets [] binary_functions [a := power] in triton {
    result = a(a, b)
  }
/-- error: power helper requires its complete copied cast-and-power return expression -/
#guard_msgs in
def poisoned (a : ℝ) : ComputeKernel :=
  with_triton_call signature call_args [a := a]
    signed_pointer_offsets [] binary_functions [power :=
      (⟨"x","exponent","return _pow(x.to(tl.float32), secret)",
        [("_pow","triton.language.extra.cuda.libdevice.pow")]⟩ : ImportedBinaryJitFunction)] in triton {
    result = power(a, b)
  }

#axiomsClean actual_value
#axiomsClean repeated_calls
end
end VeriTile.Tests.ImportedBinaryJitFunctions
