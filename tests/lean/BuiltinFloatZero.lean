import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.BuiltinFloatZero
open VeriTile.Triton

def builtin : ComputeKernel := triton { z = float(0) }
def literal : ComputeKernel := triton { z = 0.0 }

theorem exact_literal : builtin = literal := rfl

def masked (x : RegionName) : ComputeKernel := triton {
  tl.region x = tl.bfloat16
  v = tl.load(x, mask = true, other = float(0))
}

def maskedLiteral (x : RegionName) : ComputeKernel := triton {
  tl.region x = tl.bfloat16
  v = tl.load(x, mask = true, other = 0.0)
}

theorem masked_other (x : RegionName) : masked x = maskedLiteral x := rfl

/-- error: float(integer): only the exact builtin literal float(0) is supported -/
#guard_msgs in
def unsupportedInteger : ComputeKernel := triton { z = float(9007199254740993) }

/-- error: random source helper requires an explicit tl.source_helper binding -/
#guard_msgs in
def unboundAlias : ComputeKernel := triton { z = pretend_float(0) }

#axiomsClean exact_literal
#axiomsClean masked_other
end VeriTile.Tests.BuiltinFloatZero
