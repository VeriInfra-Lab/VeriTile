import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.PublicResults
import VeriTile.Meta.Specification

/-! Exact source-to-native FP32 observation. The fixed graph observer retains
the source's live accumulator and ascending scalar fma.rn.f32 contraction.
The retained numerical comparison is pointwise scalar-instruction equality;
it is not an unfused/fused rewrite, native=Real claim or finite-word restriction.
No second optimization kernel or new numerical assumption is introduced. -/

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked
open VeriTile Triton FP.Structural BmmOptimized Interface
noncomputable section

/-- All output words of every complete source grid order match the independent
padded FMA chain under the original scalar primitive correspondence. -/
theorem kernel_native (s : Shape) (c : Config) (actual target : Native.Instructions)
    (fma : ∀ l r carry,actual.fmaRN l r carry=target.fmaRN l r carry)
    (rest : Algebra Native.Word) (ps : List Program) (schedule : ps.Perm (programs s c)) :
    (nativeIO s c actual rest ps).Implements (scalarFormula s (Native.scalar target)) :=
  native_complete s c actual target fma rest ps schedule

#specNonCircular scalarFormula avoiding [Source.kernel,Execution.exec,Bridge.grid,FP.Structural.exec,runSchedule]
#stmtSurfaceSubset kernel_native ⊆
  [Shape,Config,Native.Instructions,Native.Instructions.fmaRN,Native.Word,Native.Byte,FP.Structural.Algebra,Program,
    programs,nativeIO,ProgramIO.Implements,scalarFormula,Native.scalar,
    SourceGraph.Node,SourceGraph.instInhabitedNode]
#axiomsClean kernel_native
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked
