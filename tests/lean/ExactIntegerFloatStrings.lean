import VeriTile.Triton.DSL.ExactIntegerFloatStrings
import VeriTile.Triton.Float.BitValue
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.ExactIntegerFloatStrings
open VeriTile.Triton DSL.ExactIntegerFloatStrings

def signature : CallSignature where
  pointers := [("Out",.fp32)]
  writePorts := ["Out"]

def pasted (Out : RegionName) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    x = float('-100000000.0')
    tl.store(Out, x)
    return
  }
def literal (Out : RegionName) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    x = -100000000.0
    tl.store(Out, x)
    return
  }
theorem complete_program : pasted=literal := rfl
-- These decimal strings must not silently round or lose a negative-zero sign.
#guard decimalInteger? "-100000000.0"=some "-100000000.0"
#guard decimalInteger? "0.1"=none
#guard decimalInteger? "9007199254740993"=none
#guard decimalInteger? "-0.0"=none
#guard decimalInteger? "1e8"=none
-- Concrete binary32 decoding certifies that Bloom's sentinel is exact.
theorem bloom_sentinel_exact :
    FP.Value.toRat? (FP.Value.ofBits .fp32 0xccbebc20)=some (-100000000 : Rat)  := by decide +kernel

#axiomsClean complete_program
#axiomsClean bloom_sentinel_exact
end VeriTile.Tests.ExactIntegerFloatStrings
