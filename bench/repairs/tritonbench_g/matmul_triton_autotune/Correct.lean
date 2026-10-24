import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.LogicalIO

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel
open VeriTile.Triton FP.Structural Whole
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Complete original schedule, strided FP32 inputs and FP16 output; both
matrices and the result use ordinary logical coordinates. -/
def matmulIO (g : Geometry) (ids : List Nat) :=
  Interface.logicalIO (idealAlgebra id) idealDot g ids
specification mathematical_kernel (g : Geometry) (ids : List Nat)
    (order : ids.Perm (List.range (programs g.request))) (sepA : g.C ≠ g.A) (sepB : g.C ≠ g.B) :
    VeriTile.Spec.Real (matmulIO g ids ⊨ fun ⟨a,b⟩ row col =>
      let value := ∑ k : Fin g.inner, a row k * b k col
      if g.activation == "leaky_relu" then
        if 0 ≤ value then value else (1 / 100 : ℝ) * value else value) :=
  LogicalIOProof.implements g ids order sepA sepB

#specNonCircular LogicalIOProof.target avoiding [Source.typed,Repair.runPrograms,FP.Structural.exec,FP.Structural.run]
#stmtSurfaceSubset mathematical_kernel ⊆
  [mathematical_kernel.match_1,Spec.Real,matmulIO,LogicalProgramIO.Implements,Geometry,Geometry.rows,Geometry.inner,Geometry.columns,Geometry.activation,Geometry.A,Geometry.B,Geometry.C,Geometry.request,programs,RegionName]
#axiomsClean mathematical_kernel
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel
