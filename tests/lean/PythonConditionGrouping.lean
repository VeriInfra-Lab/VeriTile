import tests.lean.PythonConditionGroupingReference
import VeriTile.Triton.DSL.PythonConditionGrouping
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.PythonConditionGrouping
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("Y",.fp32)]
  scalars := [("a",.i64),("b",.i64),("c",.i64)]
  writePorts := ["Y"]

def source (Y : RegionName) : ComputeKernel :=
  with_triton_signature signature pointer_offsets [] in triton {
    if a > 0 and b > 1 and c > 2:
        value = 11.0
    else:
        value = 13.0
    other = 17.0 if a > 0 else 19.0
    tl.store(Y, value)
    tl.store(Y + 1, other)
  }

/-- Full typed-program equality retains all runtime conditions, both branches,
source operation order, literals and stores for every input state and algebra. -/
theorem complete_program : source=PythonConditionGroupingReference.source := by rfl

open VeriTile.Triton.DSL.PythonConditionGrouping in
#guard splitWord "x and helper(\"and\", y) and z" "and" =
  ["x "," helper(\"and\", y) "," z"]
open VeriTile.Triton.DSL.PythonConditionGrouping in
#guard splitWord "candy and a[indices]" "and" = ["candy "," a[indices]"]
open VeriTile.Triton.DSL.PythonConditionGrouping in
#guard groupAnd "a and b and c and d" = .ok "(((a  and  b ) and  c ) and  d)"

/-- error: Python assignment grouping requires one complete conditional expression -/
#guard_msgs (error) in
example (Y : RegionName) : ComputeKernel :=
  with_triton_signature signature pointer_offsets [] in triton {
    value = 11.0 if a > 0
    tl.store(Y, value)
  }

#axiomsClean complete_program
end VeriTile.Tests.PythonConditionGrouping
