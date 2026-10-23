import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.NativeInterface
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.BmmChunkFwd.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open NativeGraph KernelInterface
open scoped VeriTile.Spec
noncomputable section

/-- Complete original source grid, cleanup and shared typed IO. -/
def io := BmmIO.Native.io
abbrev candidate := BmmIO.Native.candidate

/-- The original complete launch schedule, including actual cleanup launches. -/
def original (rest : Algebra Expr) (p : Parameters) (chosen : Host.Config) :=
  ProgramIO.Implementation.original (io rest p chosen).toProgramIO (fun _ => True)

/-- The existing integer-index candidate, retaining the complete schedule,
FP32 casts, K-block/MMA order, native observer and cleanup. -/
def indexCommuted (rest : Algebra Expr) (p : Parameters) (chosen : Host.Config) :=
  ProgramIO.Implementation.ofVariant (candidate rest p chosen) (fun _ => True)

/-- Both complete actual schedules are safe and succeed, their active output
cells agree, and their exterior frames are preserved. Only the integer
head-index product is commuted; no floating transformation rule is used. -/
specification bmm_chunk_fwd_execution (rest : Algebra Expr) (p : Parameters) (chosen : Host.Config) :
    original rest p chosen ≡[[]] indexCommuted rest p chosen :=
  ProgramIO.Implementation.of_equivalent (candidate rest p chosen)
    (BmmIO.Native.native_equivalent rest p chosen)

#print_fp_assumptions bmm_chunk_fwd_execution
#stmtSurfaceSubset bmm_chunk_fwd_execution ⊆
  [original,indexCommuted,Spec.Equiv,ProgramIO.Implementation,ProgramIO.Implementation.programSyntax,
   FP.Structural.Algebra,Expr,instInhabitedExpr,Parameters,Host.Config,ReadProgramIO.toProgramIO,
   io,ProgramIO.Inputs,ProgramIO.InputBindingMode.decoded,Spec.RuleEntry,Spec.ProgramSyntax.Statement]
#axiomsClean bmm_chunk_fwd_execution
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkFwd.FPEquiv
