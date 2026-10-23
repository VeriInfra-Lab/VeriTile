import bench.repairs.tritonbench_g.max_reduction.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.MaxReduction.Correct
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.LogicalIO
noncomputable section

/-- Each original stage keeps its complete PID-dependent masked vector window. -/
abbrev stage1IO := PublicIO.logicalStage1IO
abbrev stage2IO := PublicIO.logicalStage2IO

specification stage1_math_correct (inp mid : RegionName) (N B : Nat) (hB : 0<B) :
    Spec.Real (stage1IO inp mid N B ⊨ fun ⟨pid,xs⟩ _ =>
      WithBot.unbotD 0 ((Finset.univ : Finset (Fin B)).sup
        (fun i => if pid*B+i.val<N then (some (xs i) : WithBot ℝ) else ⊥))) :=
  LogicalIOProof.stage1_implements inp mid N B hB

specification stage2_math_correct (mid out : RegionName) (N B : Nat) (hB : 0<B) :
    Spec.Real (stage2IO mid out N B ⊨ fun ⟨_,xs⟩ _ =>
      WithBot.unbotD 0 ((Finset.univ : Finset (Fin B)).sup
        (fun i => if i.val<N then (some (xs i) : WithBot ℝ) else ⊥))) := by
  simpa only [LogicalIOProof.blockMaximum,Nat.zero_mul,Nat.zero_add] using
    LogicalIOProof.stage2_implements mid out N B hB

/-- Every active row of one original dimension invocation, with separately
typed maximum values and signed indices. This remains a relational judgment. -/
abbrev dimensionIO (inp ov oi : RegionName) (M N K BM p k : Nat)
    (plan : ReductionPlan (nextPower N)) :=
  PublicIO.logicalDimensionIO (KernelMath.engine Compute.real plan.tree) RealIO.observe
    inp ov oi M N K BM (nextPower N) p k

/-- Independent value/index result. Ties choose the least logical index;
the original empty-axis projection returns value zero and index zero. -/
def dimensionResult (M N BM BN p : Nat) (xs : Fin BM → Fin BN → ℝ)
    (result : (Fin BM → ℝ) × (Fin BM → Int)) : Prop :=
  let values := result.1
  let indices := result.2
  ∀ row : Fin BM, p*BM+row.val<M →
    if N=0 then values row=0 ∧ indices row=0
    else ∃ index : Nat, indices row=(index : Int) ∧ index<N ∧
      values row=PublicIO.dimensionInput xs row index ∧
      (∀ col, col<N → PublicIO.dimensionInput xs row col≤values row) ∧
      (∀ col, col<N → PublicIO.dimensionInput xs row col=values row → index≤col)

specification dimension_math_correct (inp ov oi : RegionName) (M N K BM p k : Nat)
    (hRegions : ov≠oi) (hK : 0<K) (plan : ReductionPlan (nextPower N)) :
    Spec.Real ((dimensionIO inp ov oi M N K BM p k plan).Satisfies
      (dimensionResult M N BM (nextPower N) p)) := by
  apply LogicalIOProof.dimension_implements inp ov oi M N K BM p k hRegions hK plan
  intro xs ys
  rfl

#stmtSurfaceSubset stage1_math_correct ⊆
  [Spec.Real,RegionName,stage1IO,LogicalIO.Implements,TileIndex,MaskedTileKernelIO₁.Implements,PublicIO.stage1Real,MaskedTileKernelIO₁.shape,stage1_math_correct.match_1]
#stmtSurfaceSubset stage2_math_correct ⊆
  [Spec.Real,RegionName,stage2IO,LogicalIO.Implements,TileIndex,MaskedTileKernelIO₁.Implements,PublicIO.stage2Real,MaskedTileKernelIO₁.shape,stage1_math_correct.match_1]
#stmtSurfaceSubset dimension_math_correct ⊆
  [Spec.Real,RegionName,ReductionPlan,nextPower,dimensionIO,LogicalRelationIO.Satisfies,dimensionResult,ChannelValues,IOChannel,IOChannel.mk,TileDType.real,TileDType.int,ProgramIO.Satisfies,PublicIO.dimensionIO,KernelMath.engine,Compute.real,ReductionPlan.tree,RealIO.observe]
#axiomsClean stage1_math_correct
#axiomsClean stage2_math_correct
#axiomsClean dimension_math_correct
#specNonCircular dimensionResult avoiding [Source.Engine.algebra,Source.rowValue,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MaxReduction.Correct
