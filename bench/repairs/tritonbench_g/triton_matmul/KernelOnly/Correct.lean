import bench.repairs.tritonbench_g.triton_matmul.KernelOnly.Proofs.LogicalIO

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly
open VeriTile.Triton FP.Structural FPValue
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Finite M×K/K×N logical inputs with the original strides and complete M×N
output. cellType is FP16 for half and E4M3 for e4m3. The selected source retains
its FP32 accumulator and final storage cast; Real idealizes those operations.
The complete program order is explicit, including grouped tiles and masked tails. -/
def realIO (g : Geometry) (ids : List Nat) : ProgramIO ℝ ℝ :=
  PublicIO.io g KernelRealModel.ideal KernelRealModel.idealDot ids

/-- Independent full-K GEMM on the supplied logical input matrices. -/
def mathematical (g : Geometry) (ids : List Nat) :
    (realIO g ids).Inputs → (realIO g ids).Outputs :=
  PublicIO.expected g (fun a b i j => ∑ k : Fin g.inner,a i k*b k j)

/-- Every complete scheduling order is safe, succeeds, produces the full
mathematical matrix and preserves all cells outside the output window. -/
specification kernel_real (g : Geometry) (ids : List Nat)
    (order : ids.Perm (List.range (programs g.request))) (sepA : g.C≠g.A) (sepB : g.C≠g.B) :
    VeriTile.Spec.Real (PublicIO.logicalRealIO g ids ⊨ fun xs i j => ∑ k : Fin g.inner, xs.1 i k * xs.2 k j) := by
  apply (PublicIO.logical_iff g ids (fun a b i j => ∑ k : Fin g.inner,a i k*b k j)).2
  exact PublicIO.implements g KernelRealModel.ideal KernelRealModel.idealDot KernelRealModel.idealNumerics
    (KernelRealModel.ideal_primitives g.storage) ids order sepA sepB
    (fun a b i j => ∑ k : Fin g.inner,a i k*b k j)
    (KernelRealModel.ideal_reference g.request)

#specNonCircular mathematical avoiding [Source.typed,runPrograms,FP.Structural.exec,FP.Structural.runSchedule]
#axiomsClean kernel_real
#stmtSurfaceSubset kernel_real ⊆ [VeriTile.Spec.Real,PublicIO.logicalRealIO,LogicalProgramIO.Implements,
  RegionName,Geometry,Geometry.request,Geometry.C,Geometry.A,Geometry.B,
  Geometry.rows,Geometry.inner,Geometry.columns,programs,PublicIO.MatrixInputs]
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly
