import bench.tritonbench_g.attention_llama.Proofs.Source
import bench.tritonbench_g.attention_llama.Proofs.RealScalarInvariant

/-! Exact Real-semantics connection from the selected runtime-scalar source to
the existing literal-scale proof. Both sides run from the same bound state.
The comparison preserves full successor states, failure, and every executed
memory-access safety obligation. No native arithmetic agreement is assumed. -/
namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealSourceTransport
open VeriTile Triton
open RealScalarInvariant
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- The existing mathematical proof's source, used only as a transport target.
The public program remains `Source.kernel`, with its runtime scale register. -/
def literalSource (c : Source.Config) (scale : ℝ) : ComputeKernel :=
  (if c.fp8 then AttentionLlama.TypedSource.repairedFP8
    else AttentionLlama.TypedSource.repaired)
    c.Q c.K c.V c.Out scale
    c.sqz c.sqh c.sqm c.sqk c.skz c.skh c.skn c.skk
    c.svz c.svh c.svk c.svn c.soz c.soh c.som c.son
    c.NH c.H c.N c.SP 64 64 c.D c.causal

/-- Real projection is supported for the actual selected source, before any
runtime execution or scalar-value premise is considered. -/
theorem projection_supported (c : Source.Config) :
    (Source.kernel c).eraseDType.toAlgorithm? =
      .ok (Source.kernel c).eraseDType.toAlgKernel := by
  have projected : (Source.kernel c).toAlgorithm? = .ok (Source.kernel c).toAlgKernel := by
    cases hf : c.fp8 <;> simp only [Source.kernel,hf,Bool.false_eq_true,ite_false,ite_true] <;> rfl
  simp only [ComputeKernel.eraseDType,projected]
  simp

private theorem source_relation (c : Source.Config) (scale : ℝ) :
    Block scale (Source.kernel c).eraseDType.toAlgKernel.body
      (literalSource c scale).eraseDType.toAlgKernel.body := by
  cases hf : c.fp8
  all_goals
    simp only [Source.kernel,literalSource,hf,Bool.false_eq_true,ite_false,ite_true,
      TypedSource.half,TypedSource.bytes,
      AttentionLlama.TypedSource.repaired,AttentionLlama.TypedSource.repairedFP8,
      ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
      Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,VeriTile.Triton.eraseDType,
      NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType,
      Except.bind,Bind.bind,List.map_cons,List.map_nil,List.drop_succ_cons,List.drop_zero]
    simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType,FloatDType.toTileDType,NumericDType.eraseDType,
      ComparableDType.eraseDType,IntegralDType.eraseDType,List.map_cons,List.map_nil]
    repeat first
      | exact Block.nil scale
      | apply Block.cons
      | exact Clause.assign scale _ _ _ _ (by decide)
      | exact Clause.scale scale _ _
      | apply Clause.conditional
      | apply Clause.dynamicRange _ (by decide)
      | exact Clause.ptrStore scale _ _ _ _ _

/-- Exact execution, including failure, of the actual runtime source. Equality
of entire successor states also transports every raw-cell frame property. -/
theorem execution_eq (c : Source.Config) (scale : ℝ) (s : BlockState)
    (bound : Bound scale s) :
    exec (Source.kernel c).eraseDType.toAlgKernel s =
      exec (literalSource c scale).eraseDType.toAlgKernel s :=
  (source_relation c scale).execution s bound

/-- Access safety follows the same actual successors on both source paths,
including dynamic loop bounds, causal branches, masks and byte loads. -/
theorem traceSafe_iff (c : Source.Config) (scale : ℝ) (bounds : RegionBounds)
    (s : BlockState) (bound : Bound scale s) :
    Kernel.TraceSafe bounds (Source.kernel c).eraseDType.toAlgKernel s ↔
      Kernel.TraceSafe bounds (literalSource c scale).eraseDType.toAlgKernel s :=
  (source_relation c scale).safety bounds s bound

theorem bound_preserved (c : Source.Config) (scale : ℝ) (s final : BlockState)
    (bound : Bound scale s)
    (ran : exec (Source.kernel c).eraseDType.toAlgKernel s=some final) :
    Bound scale final := by
  rw [execution_eq c scale s bound] at ran
  exact (source_relation c scale).preserved s final bound ran

#axiomsClean execution_eq
#axiomsClean projection_supported
#axiomsClean traceSafe_iff
#axiomsClean bound_preserved
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealSourceTransport
