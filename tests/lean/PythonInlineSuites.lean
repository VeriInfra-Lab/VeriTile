import VeriTile.Triton.DSL.PythonInlineSuites
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.PythonInlineSuites
open VeriTile.Triton

#guard DSL.PythonInlineSuites.normalize "if flag: x = 1" == "if flag { x = 1 }"
#guard DSL.PythonInlineSuites.normalize "if mode == \"a:b\": x = 1" == "if mode == \"a:b\" { x = 1 }"
#guard DSL.PythonInlineSuites.normalize "if xs[0:2]: x = 1" == "if xs[0:2] { x = 1 }"
#guard DSL.PythonInlineSuites.normalize "x = \"if a: b\"" == "x = \"if a: b\""
#guard DSL.PythonInlineSuites.normalize "if flag:\n    x = 1" == "if flag:\n    x = 1"
#guard DSL.PythonInlineSuites.normalize "elif flag: x = 2" == "elif flag { x = 2 }"
#guard DSL.PythonInlineSuites.normalize "else: x = 3" == "else { x = 3 }"

#guard DSL.PythonInlineSuites.normalize "\"\"\"\nif a: b\n\"\"\"" == "\"\"\"\nif a: b\n\"\"\""

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  constexpr := ["B","ENABLED"]

def copied (enabled : Bool) (X Y : RegionName) (B : Nat) : ComputeKernel :=
  let ENABLED := enabled
  with_triton_signature signature in triton {
    offsets = tl.arange(0, B,)
    x = tl.load(X + offsets,)
    if ENABLED: x = x * 2.0
    if not ENABLED: x = x + 1.0
    tl.store(Y + offsets, x,)
  }

/-- Both source conditions remain in order. The ordinary constexpr machinery
selects the same complete typed program for every caller flag and shape. -/
theorem complete_program (enabled : Bool) (X Y : RegionName) (B : Nat) :
    copied enabled X Y B =
      (let ENABLED := enabled
       with_triton_signature signature in triton {
         offsets = tl.arange(0, B)
         x = tl.load(X + offsets)
         if ENABLED { x = x * 2.0 }
         if not ENABLED { x = x + 1.0 }
         tl.store(Y + offsets, x)
       }) := rfl

def branches (enabled : Bool) (X Y : RegionName) (B : Nat) : ComputeKernel :=
  let ENABLED := enabled
  with_triton_signature signature in triton {
    offsets = tl.arange(0, B)
    x = tl.load(X + offsets)
    if ENABLED: x = x * 2.0  # keep the original Python comment
    else: x = x + 1.0
    tl.store(Y + offsets, x)
  }

theorem complete_branches (enabled : Bool) (X Y : RegionName) (B : Nat) :
    branches enabled X Y B =
      (let ENABLED := enabled
       with_triton_signature signature in triton {
         offsets = tl.arange(0, B)
         x = tl.load(X + offsets)
         if ENABLED { x = x * 2.0 } else { x = x + 1.0 }
         tl.store(Y + offsets, x)
       }) := rfl

#guard DSL.PythonInlineSuites.escapeNames "partial = partial + 1" == "«partial» = «partial» + 1"
#guard DSL.PythonInlineSuites.escapeNames "x = \"partial\"" == "x = \"partial\""
#guard DSL.PythonInlineSuites.escapeNames "# partial\n«partial» = 1" == "# partial\n«partial» = 1"

def reserved (X Y : RegionName) (B : Nat) : ComputeKernel :=
  let ENABLED := Bool.true
  with_triton_signature signature in triton {
    partial = tl.load(X + tl.arange(0, B))
    tl.store(Y + tl.arange(0, B), partial)
  }

theorem original_identifier (X Y : RegionName) (B : Nat) : reserved X Y B =
    (let ENABLED := Bool.true
     with_triton_signature signature in triton {
       «partial» = tl.load(X + tl.arange(0, B))
       tl.store(Y + tl.arange(0, B), «partial»)
     }) := rfl

#axiomsClean original_identifier

#axiomsClean complete_program
#axiomsClean complete_branches
end VeriTile.Tests.PythonInlineSuites
