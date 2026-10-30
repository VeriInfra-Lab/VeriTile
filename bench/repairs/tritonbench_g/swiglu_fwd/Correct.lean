import bench.repairs.tritonbench_g.swiglu_fwd.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.SwigluFwd
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluFwd.Whole
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- The selected row/tile of the actual FP32 source, with complete logical
input snapshots and every declared output lane. -/
abbrev swigluIO := Logical.io

specification swiglu_kernel_correct (c : Layout) (B row tile : Nat) :
    VeriTile.Spec.Real (swigluIO c B row tile ⊨ fun xs i =>
      xs.x i * xs.y i / (1 + Real.exp (-xs.x i))) :=
  Logical.reader_implements c B row tile

#axiomsClean swiglu_kernel_correct
#stmtSurfaceSubset swiglu_kernel_correct ⊆
  [VeriTile.Spec.Real,Layout,swigluIO,LogicalProgramIO.Implements,Logical.Inputs,Logical.Inputs.x,Logical.Inputs.y]
end
end VeriTile.Bench.Repairs.SwigluFwd
