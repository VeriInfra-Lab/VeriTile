import VeriTile.Triton.DSL.FP32CallValues
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
open VeriTile.Triton
namespace VeriTile.Tests.FP32CallValues
noncomputable section
def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.boolean)]
  scalars := [("threshold",.fp32)]
  readPorts := ["X"]
  writePorts := ["Y"]
def pasted (X Y : RegionName) (threshold : ℝ) : ComputeKernel :=
  with_triton_call signature call_args [threshold := threshold] integer_args [] arange_args [] in triton {
    keep = tl.load(X) > threshold
    tl.store(Y, keep)
  }

def expanded (X Y : RegionName) (threshold : ℝ) : ComputeKernel :=
  .mk [X] [Y]
    [.assign .real [] "__veritile_load_0"
      (.compute (.load .fp32 (.region X (.constNat 0)) .none)),
     .assign .bool [] "keep" (.alg (.gt .real .nil (.ref .real [] "__veritile_load_0")
       (.withPrecision .real .fp32 (.const threshold)))),
     .store .bool [] (.region Y.cast (.constNat 0))
       (.alg (.ne .int .nil (.boolToInt (.ref .bool [] "keep")) (.constInt 0))) .none]
theorem complete_program : pasted=expanded := rfl

/-- The actual threshold decoder is FP32 under every caller precision and
arbitrary literal/comparison callbacks; no numerical identity is assumed. -/
theorem threshold_precision {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (pr : Option ComputeDType) (s : FP.Structural.State α) (p : ℝ) :
    FP.Structural.evalOp M pr (.withPrecision .real .fp32 (.const p)) s =
      some (fun _ => M.literal (some .fp32) .real p) := rfl

/-- error: FP32 call values must remain immutable -/
#guard_msgs (error) in
example (X Y : RegionName) (threshold : ℝ) : ComputeKernel :=
  with_triton_call signature call_args [threshold := threshold] integer_args [] arange_args [] in triton {
    threshold -= 1.0
    tl.store(Y, True)
  }

#axiomsClean complete_program
#axiomsClean threshold_precision
end
end VeriTile.Tests.FP32CallValues
