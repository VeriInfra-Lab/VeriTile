import tests.lean.PythonShapeTupleReference
import VeriTile.Triton.DSL.PythonShapeTuples
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.PythonShapeTuples
open VeriTile.Triton PythonShapeTupleReference
noncomputable section

def original (Out : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    values = tl.full((B,), value=0, dtype=Out.dtype.element_ty)
    tl.store(Out + tl.arange(0, B), values)
  }

theorem complete_program : original=reference := rfl
#guard DSL.PythonShapeTuples.normalize "tl.full((N), value=0)" == "tl.full((N), value=0)"
#guard DSL.PythonShapeTuples.normalize "x = (N,) # tl.full((N,), 0)" == "x = (N,) # tl.full((N,), 0)"
#guard DSL.PythonShapeTuples.normalize "x = 'tl.full((N,), 0)'" == "x = 'tl.full((N,), 0)'"
#axiomsClean complete_program
end
end VeriTile.Tests.PythonShapeTuples
