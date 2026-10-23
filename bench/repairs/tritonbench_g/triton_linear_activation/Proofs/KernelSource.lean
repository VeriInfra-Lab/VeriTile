import bench.repairs.tritonbench_g.triton_linear_activation.SourceKernels

namespace VeriTile.Bench.Repairs.TritonLinearActivation.KernelSource
open VeriTile.Triton FP.Structural
noncomputable section
def front (mode : Mode) (p : Parameters) := (source mode p).surfaceBody.take 17
def statement (mode : Mode) (p : Parameters) : ComputeStmt :=
  ((source mode p).surfaceBody[17]?).getD (.effectMarker "missing loop")
def body (mode : Mode) (p : Parameters) : List ComputeStmt :=
  match statement mode p with
  | .forRangeInt _ _ _ _ body => body
  | _ => []
def tailCode (mode : Mode) (p : Parameters) := (source mode p).surfaceBody.drop 18

theorem source_split (mode : Mode) (p : Parameters) :
    (source mode p).surfaceBody=front mode p++[statement mode p]++tailCode mode p := by rfl

theorem source_statements (mode : Mode) (p : Parameters) : (source mode p).surfaceBody.length=26 := by rfl

#axiomsClean source_split
#axiomsClean source_statements
#guard_msgs (drop info) in
#auditModuleAxioms

end
end VeriTile.Bench.Repairs.TritonLinearActivation.KernelSource
