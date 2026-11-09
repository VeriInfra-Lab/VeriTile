import VeriTile.Triton.DSL.PythonFullConstructors
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.PointerElementLoadCast
open VeriTile.Triton
noncomputable section

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp16)]
  readPorts := ["X","Y"]
  writePorts := ["Y"]

def source (X Y : RegionName) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    acc = tl.load(Y)
    bias = tl.load(X).to(Y.dtype.element_ty)
    acc = acc + bias
    tl.store(Y, acc)
  }

/-- The cast precedes addition: neither FP32 addition followed by a store
conversion nor a bare FP32 load is the source's FP16 bias addition. -/
def expected (X Y : RegionName) : ComputeKernel := .mk [X,Y] [Y] [
  .assign .fp16 [] "acc" (.alg (.load .fp16 (.region (Region.cast Y) (.constNat 0)) .none)),
  .assign .real [] "__veritile_load_0" (.compute (.load .fp32 (.region X (.constNat 0)) .none)),
  .assign .fp16 [] "bias" (.alg (.castFloat .real .fp16
    (.withPrecision .real .fp32 (.ref .real [] "__veritile_load_0")))),
  .assign .fp16 [] "acc" (.alg (.add .fp16 .nil (.ref .fp16 [] "acc") (.ref .fp16 [] "bias"))),
  .store .fp16 [] (.region (Region.cast Y) (.constNat 0)) (.alg (.ref .fp16 [] "acc")) .none]

theorem complete_program (X Y : RegionName) : source X Y = expected X Y := rfl

#axiomsClean complete_program
#auditModuleAxioms
end
end VeriTile.Tests.PointerElementLoadCast
