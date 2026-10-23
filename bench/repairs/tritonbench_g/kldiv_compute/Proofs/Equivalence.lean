import bench.repairs.tritonbench_g.kldiv_compute.IO
import bench.repairs.tritonbench_g.kldiv_compute.Proofs.Program

namespace VeriTile.Bench.Repairs.KldivCompute.EquivalenceProof
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

/-- Both typed quotations retain the same floating suffix. Only natural-number
address multiplication is commuted, for an arbitrary floating backend. -/
theorem execution_eq {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    exec M (KernelDSL.kernel p) s = exec M (KernelDSL.commutedKernel p) s := by
  let typed := VeriTile.Bench.TritonBenchG.KldivCompute.TypedSource.kernel
    .fp32 .fp32 .fp32
  change run M (typed Bool.false p.x p.y p.output p.n p.block).surfaceBody s =
    run M (typed Bool.true p.x p.y p.output p.n p.block).surfaceBody s
  rw [VeriTile.Bench.TritonBenchG.KldivCompute.FPProof.decomposition .fp32 .fp32 .fp32 Bool.false,
    VeriTile.Bench.TritonBenchG.KldivCompute.FPProof.decomposition .fp32 .fp32 .fp32 Bool.true,
    run_append, run_append]
  apply congrArg (fun (result : Option (State α)) => result.bind
    (run M ((typed Bool.false p.x p.y p.output p.n p.block).surfaceBody.drop 2)))
  have index_eq (t : State α) :
      step M (VeriTile.Bench.TritonBenchG.SinKernel.FPProof.indexStmt Bool.false p.block) t =
      step M (VeriTile.Bench.TritonBenchG.SinKernel.FPProof.indexStmt Bool.true p.block) t := by
    simpa only [VeriTile.Bench.TritonBenchG.SinKernel.FPProof.indexStmt,
      Bool.false_eq_true, ↓reduceIte, step, FP.Structural.evalExpr,
      evalOp_unfold, numeric, Option.bind] using
      VeriTile.Bench.TritonBenchG.SinKernel.FPProof.index_step M t p.block
  simp only [VeriTile.Bench.TritonBenchG.SinKernel.FPProof.sourcePrefix, run, index_eq]

/-- Successful actual executions, active output equality, and the two masked
memory frames. No floating algebraic law or value-domain restriction is used. -/
theorem equivalent (p : Parameters) :
    MaskedIO₂Equiv (FloatingIO.original p) (FloatingIO.commuted p) := by
  refine ⟨by simp [MaskedPrivateScratch, FloatingIO.original],
    by simp [MaskedPrivateScratch, FloatingIO.commuted, FloatingIO.original,
      MaskedKernelIO₂.withKernel], ?_⟩
  intro α _ M s
  obtain ⟨final, executed, memory⟩ := Program.complete_correct M p s
  have left : exec M (KernelDSL.kernel p) s = some final := executed
  have right : exec M (KernelDSL.commutedKernel p) s = some final := by
    rw [← execution_eq]
    exact left
  have frame : MaskedFrame (FloatingIO.original p) s final := by
    intro r o outside _
    rw [memory]
    apply Program.expected_frame
    rcases outside with different | outside
    · exact Or.inl different
    · right
      intro i active
      exact outside i.1 active
  exact ⟨final, final, left, right, fun _ _ => rfl, frame, frame⟩

#axiomsClean execution_eq
#axiomsClean equivalent
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivCompute.EquivalenceProof
