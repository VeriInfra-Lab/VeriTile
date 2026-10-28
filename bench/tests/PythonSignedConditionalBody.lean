import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.PythonSignedConditionalBody
open VeriTile.Triton

def signature : CallSignature where
  constexpr := ["T", "K", "FLAG", "ZERO"]

def source (T K : Int) (FLAG : Bool) : ComputeKernel :=
  let ZERO : Int := 0
  with_triton_call signature call_args [] in triton {
    value = (T - 1) * K if FLAG else 0
  }

def typedZero (T K : Int) (FLAG : Bool) : ComputeKernel :=
  let ZERO : Int := 0
  with_triton_call signature call_args [] in triton {
    value = (T - 1) * K if FLAG else ZERO
  }

/-- Zero acquires the signed branch carrier, including T = 0. It is never
interpreted as an FP operation or a saturating natural subtraction. -/
theorem signed_zero (T K : Int) (FLAG : Bool) :
    source T K FLAG = typedZero T K FLAG := rfl

#axiomsClean signed_zero
end VeriTile.Tests.PythonSignedConditionalBody
