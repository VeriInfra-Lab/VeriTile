/- Auxiliary single-source native correctness. No program-pair equivalence is claimed. -/
import bench.repairs.tritonbench_g.matmul_leakyrelu.KernelOnly.Proofs.IOComplete
import VeriTile.Meta.Specification

/-! Exact source-to-independent native recurrence observation. The six existing
primitive laws describe the selected contracted numerical model; this is not
an equivalence with the separate-dot/add source or an experimental premise. -/
namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly
open VeriTile.Triton FP.Structural Interface
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu
noncomputable section
local instance nativeWordInhabited : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Ordered K32 dot accumulation, exact activation branch and FP16 conversion,
over all admitted words and every complete source launch order. -/
theorem matmul_leakyrelu_native_observation (M : Algebra Word) (D : FPFused.DotAcc Word)
    (Q : Whole.Contracted.Numerics Word) (law : Whole.Contracted.PrimitiveLaws M D Q)
    (p : KernelInterface.Parameters) (separateA : p.C≠p.A) (separateB : p.C≠p.B)
    (order : Schedule p.request) :
    (io M D p order).Implements (nativeReference Q p) :=
  IOComplete.native_implements M D Q law p separateA separateB order

#specNonCircular nativeReference avoiding [FP.Structural.exec,run,dsl,sourceDsl]
#stmtSurfaceSubset matmul_leakyrelu_native_observation ⊆
  [Word,nativeWordInhabited,RegionName,FP.Structural.Algebra,FPFused.DotAcc,
    Whole.Contracted.Numerics,Whole.Contracted.PrimitiveLaws,
    KernelInterface.Parameters,KernelInterface.Parameters.C,KernelInterface.Parameters.A,
    KernelInterface.Parameters.B,KernelInterface.Parameters.request,Schedule,
    io,ProgramIO.Implements,nativeReference]
#axiomsClean matmul_leakyrelu_native_observation
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly
