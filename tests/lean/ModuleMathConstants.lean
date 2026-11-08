import VeriTile.Triton.DSL.ModuleMathConstants
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.ModuleMathConstants
open VeriTile.Triton
noncomputable section

def profile : CallSignature where
  pointers := [("input",.fp32),("output",.fp32)]
  sourceConstants := [("sqrt2","math.sqrt(2.0)"),("sqrt2pi","math.sqrt(2.0 / math.pi)")]
  readPorts := ["input"]
  writePorts := ["output"]
def bound (input output : RegionName) : ComputeKernel :=
  with_triton_call profile call_args [] in triton {
    value = tl.load(input)
    value = sqrt2pi * value / sqrt2
    tl.store(output, value)
  }
def directProfile : CallSignature where
  constexpr := ["sqrt2","sqrt2pi"]
  pointers := [("input",.fp32),("output",.fp32)]
  readPorts := ["input"]
  writePorts := ["output"]
def direct (input output : RegionName) (sqrt2 sqrt2pi : ℝ) : ComputeKernel :=
  with_triton_call directProfile call_args [] in triton {
    value = tl.load(input)
    value = sqrt2pi * value / sqrt2
    tl.store(output, value)
  }
def tags (input output : RegionName) : ComputeKernel :=
  direct input output (Real.sqrt 2.0) (Real.sqrt (2.0/Real.pi))

/-- Exact IR/source-tag binding only; this asserts no host rounding equality. -/
theorem source_tags_exact (input output : RegionName) : bound input output=tags input output := by rfl
#axiomsClean source_tags_exact

/-- error: module math constants cannot be reassigned -/
#guard_msgs in
example (input output : RegionName) : ComputeKernel :=
  with_triton_call profile call_args [] in triton {
    sqrt2 = 0.0
    tl.store(output, tl.load(input))
  }

def unbound : CallSignature where
  pointers := [("input",.fp32),("output",.fp32)]
  sourceConstants := [("factor","math.sqrt(caller_value)")]
/-- error: module math binding requires a closed numeric/sqrt/pi expression -/
#guard_msgs in
example (input output : RegionName) : ComputeKernel :=
  with_triton_call unbound call_args [] in triton {
    tl.store(output, tl.load(input) * factor)
  }
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.ModuleMathConstants
