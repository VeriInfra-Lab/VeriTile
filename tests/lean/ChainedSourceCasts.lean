import tests.lean.ChainedSourceCastsReference
import VeriTile.Triton.DSL.ChainedSourceCasts
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.ChainedSourceCasts
open VeriTile.Triton ChainedSourceCastsReference

def copied (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    off = tl.arange(0, B)
    x = tl.load(X + off)
    y = x.to(Y.dtype.element_ty).to(tl.float32)
    tl.store(X + off, y)
  }

/-- Whole typed-program equality checks the intermediate FP16 conversion and
its FP32 successor against an independently elaborated parenthesized program. -/
theorem complete_typed_program (X Y : RegionName) (B : Nat) :
    copied X Y B=expected X Y B := rfl

#axiomsClean complete_typed_program
end VeriTile.Tests.ChainedSourceCasts
