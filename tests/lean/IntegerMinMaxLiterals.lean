import tests.lean.IntegerMinMaxReference
import VeriTile.Triton.DSL.IntegerMinMaxLiterals
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.IntegerMinMaxLiterals
open VeriTile.Triton IntegerMinMaxReference

def copied (I F Out : RegionName) (window B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [window := window] in triton {
    offsets = tl.arange(0, B)
    x = tl.load(I + offsets)
    y = tl.maximum(x - window, 0)
    tl.store(Out + offsets, y)
  }

def floatingCopy (I F Out : RegionName) (window B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [window := window] in triton {
    offsets = tl.arange(0, B)
    x = tl.load(I + offsets)
    x = tl.load(F + offsets)
    y = tl.maximum(x, 0)
    tl.store(Out + offsets, y.to(tl.int32))
  }

/-- The complete typed integer load/max/store program agrees with the
independent integer-literal reference, including its output conversion. -/
theorem integer_program (I F Out : RegionName) (window B : Nat) :
    copied I F Out window B=integer I F Out window B := by rfl

/-- Rebinding the metadata name to a floating load keeps floating maximum
and its original weak literal, including the selected floating callback. -/
theorem floating_program (I F Out : RegionName) (window B : Nat) :
    floatingCopy I F Out window B=floating I F Out window B := by rfl

#axiomsClean integer_program
#axiomsClean floating_program
end VeriTile.Tests.IntegerMinMaxLiterals
