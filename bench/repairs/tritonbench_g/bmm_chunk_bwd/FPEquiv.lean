import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.KernelRewrite

namespace VeriTile.Bench.Repairs.BmmChunkBwd.FPEquiv
open VeriTile Triton FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
open scoped VeriTile.Spec

/-- Exact integer head-index commutation; no floating numerical rewrite law. -/
def assumptions : Spec.Assumptions FP.GuardedFragment := []
def original (h : HostCall) (choice : Fin 9) : Program := KernelRewrite.program (Kernels.fp32Source h (tuning choice))
def candidate (h : HostCall) (choice : Fin 9) : Program := KernelRewrite.program (Kernels.indexCommuted h (tuning choice))

/-- The entire quoted FP32 program is equivalent under the original arbitrary
floating algebra. Only integer multiplication order in a head index changes.
KernelInterface.versions_eq composes this exact source result over the full
original grid; its native theorem retains physical output/Good/frame scope. -/
specification bmm_chunk_bwd_equiv (h : HostCall) (choice : Fin 9) :
    original h choice ≡[assumptions] candidate h choice :=
  KernelRewrite.equivalent h (tuning choice)

#print_fp_assumptions bmm_chunk_bwd_equiv
#stmtSurfaceSubset bmm_chunk_bwd_equiv ⊆
  [HostCall,TypedSource.HostCall,original,candidate,assumptions,Spec.Equiv,
   Program,instProgramSyntaxProgram]
#axiomsClean bmm_chunk_bwd_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.BmmChunkBwd.FPEquiv
