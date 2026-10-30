import bench.tritonbench_g.decay_cumsum.Proofs.BackwardExecution
import bench.tritonbench_g.decay_cumsum.Proofs.BackwardWitness
import bench.tritonbench_g.decay_cumsum.Proofs.TypedExecution
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.ReverseRegression
open VeriTile.Triton TypedSource
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

/-- All 18 source/storage variants retain the signed reverse loop. Its lower
bound is computed in Int, before any natural-number truncation. -/
theorem signed_source (source gate : Storage) (swapped : Bool)
    (dqi dqo dki dko q k g dg : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) :
    ((backward source gate swapped dqi dqo dki dko q k g dg
      sH sT sD B H T BT BK DK scale).eraseDType.toAlgKernel.body).drop 12 =
      [BackwardExecution.sourceLoop DK BT BK] := by
  cases source <;> cases gate <;> cases swapped
  all_goals simp only [backward,BackwardExecution.sourceLoop,BackwardLoop.sourceBody,
    RealReference.bwdIterBodyG,Bool.false_eq_true,↓reduceIte,ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,ComputeDType.eraseDType,
    List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,eraseDType_float]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType]
  all_goals rfl

/-- At BT=0 the actual reverse source executes no iteration. In particular
it never uses the historical Nat reference's one-iteration zero-size case. -/
theorem zero_iterations (source gate : Storage) (swapped : Bool)
    (dqi dqo dki dko q k g dg : RegionName)
    (sH sT sD B H T BK DK : Nat) (scale : ℝ) (u : BlockState) :
    stepStmts (((backward source gate swapped dqi dqo dki dko q k g dg
      sH sT sD B H T 0 BK DK scale).eraseDType.toAlgKernel.body).drop 12) u = some u := by
  rw [signed_source]
  simp [BackwardExecution.singleton,BackwardExecution.sourceLoop,stepStmt,evalOp,
    NumericDType.sub,Tile.bop,Int.ofNat_eq_natCast]

/-- Two actual reverse iterations, masked lanes, nonzero program IDs and finite
storage execute for every independent data/gate storage pair and candidate. -/
theorem complete_finite_execution (source gate : Storage) (swapped : Bool) :
    ∃ final, exec (BackwardWitness.allocation.flattenKernel
      (backward source gate swapped "dqi" "dqo" "dki" "dko" "q" "k" "g" "dg"
        16 3 1 2 2 4 2 2 3 1).eraseDType.toAlgKernel)
      (BackwardWitness.allocation.flattenState ForwardWitness.initial) = some final :=
  BackwardWitness.executes source gate swapped

/-- The complete typed reverse program also succeeds, from empty registers. -/
theorem complete_typed_execution :
    FP.Structural.exec (FP.ProfiledRewrite.engine TypedExecution.model)
      TypedExecution.backwardKernel TypedExecution.initial =
      some TypedExecution.backwardExecution.val := TypedExecution.backwardExecution.property

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DecayCumsum.ReverseRegression
