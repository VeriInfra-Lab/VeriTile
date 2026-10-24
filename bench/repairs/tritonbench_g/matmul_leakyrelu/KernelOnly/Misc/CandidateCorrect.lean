import bench.repairs.tritonbench_g.matmul_leakyrelu.KernelOnly.Proofs.IOComplete
import VeriTile.Meta.Specification

/-! Optional candidate correctness: independent mathematical matrix product for every legal complete grid order of Whole.contractedKernel.
The public IO declares FP32 strided A/B channels and the FP16 C window. -/
namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly
open VeriTile.Triton FP.Structural Interface
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu
noncomputable section

/-- Complete contracted candidate, independent matrix formula, actual access safety,
all output cells and the exact exterior frame. Read-only inputs may alias. -/
theorem matmul_leakyrelu_correctness (p : KernelInterface.Parameters)
    (separateA : p.C≠p.A) (separateB : p.C≠p.B) (order : Schedule p.request) :
    Spec.Real ((io (Whole.idealAlgebra id) Whole.Contracted.idealDotAcc p order).Implements
      (realReference p)) :=
  IOComplete.real_implements p separateA separateB order

#specNonCircular realReference avoiding [FP.Structural.exec,run,dsl,sourceDsl]
#stmtSurfaceSubset matmul_leakyrelu_correctness ⊆
  [Spec.Real,RegionName,KernelInterface.Parameters,KernelInterface.Parameters.C,
    KernelInterface.Parameters.A,KernelInterface.Parameters.B,KernelInterface.Parameters.request,
    Schedule,io,Whole.idealAlgebra,Whole.Contracted.idealDotAcc,ProgramIO.Implements,realReference]
#axiomsClean matmul_leakyrelu_correctness
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly
