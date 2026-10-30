import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.JITReaderCorrect

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Correct
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- The least logical index attaining the maximum of all n Real values.
This independent definition exposes both maximality and the tie convention. -/
def firstMaximum (n : Nat) (positive : 0 < n) (xs : Fin n → ℝ) : Fin n :=
  Classical.choose (show ∃ i : Fin n,
    (∀ j, xs j ≤ xs i) ∧ (∀ j, xs j = xs i → i.val ≤ j.val) from
    ⟨Math.argmax n positive xs, Math.argmax_spec n positive xs⟩)

/-- Complete first grid and second quotation. The existing integer/bool
conversion-macro idealization uses projectedKind; it is not finite-clamp identity.
Requested precision is erased only in Real semantics; both scratch windows and the scalar output are explicit
in JITIO.realGlobal. The original region aliases and coverage are retained. -/
abbrev globalIO (kind : Request.Kind) (rest : Algebra Math.Value)
    (X V : RegionName) (I O : Region .int) (c : GlobalConfig) (index64 : Bool) :=
  JITIO.logicalGlobal (JITIO.projectedKind kind) rest X V I O c index64

/-- Independent first-maximum index over the complete logical Real input. -/
specification global (kind : Request.Kind) (rest : Algebra Math.Value)
    (X V : RegionName) (I O : Region .int) (inputValue : X ≠ V) (inputIndex : X ≠ Region.cast I)
    (scratchApart : V ≠ Region.cast I) (c : GlobalConfig) (coverage : Math.GlobalCoverage c)
    (index64 : Bool) :
    Spec.Real (globalIO kind rest X V I O c index64 ⊨ fun xs =>
      Int.ofNat (firstMaximum c.size coverage.sizePositive xs).val) := by
  exact JITReaderCorrect.global kind rest X V I O inputValue inputIndex scratchApart c coverage index64
    (firstMaximum c.size coverage.sizePositive)
    (JITReaderCorrect.firstMaximum_eq c.size coverage.sizePositive)

/-- The actual dimension quotation loops over the whole reduction axis and
writes only active rows. This is a separate source branch, with its original
arbitrary block dimensions, base and input/output alias scope. -/
specification dimension (rest : Algebra Math.Value) (X : RegionName) (O : Region .int)
    (M N K BM BN base pid : Nat) (index64 : Bool) (positive : 0 < BN)
    (nonempty : 0 < N) (innerPositive : 0 < K) :
    Spec.Real (JITIO.logicalDimension rest X O M N K BM BN base pid index64 ⊨ fun xs j =>
      Int.ofNat (firstMaximum N nonempty (JITIO.dimensionRow M N K BM base pid xs j)).val) := by
  exact JITReaderCorrect.dimension rest X O M N K BM BN base pid index64 positive nonempty innerPositive
    (firstMaximum N nonempty) (JITReaderCorrect.firstMaximum_eq N nonempty)

#stmtSurfaceSubset global ⊆
  [Request.Kind,FP.Structural.Algebra,Math.Value,RegionName,Region,TileDType.int,TileDType.real,Region.cast,
   GlobalConfig,Math.GlobalCoverage,Spec.Real,globalIO,LogicalProgramIO.Implements,
   firstMaximum,GlobalConfig.size,Math.GlobalCoverage.sizePositive]
#stmtSurfaceSubset dimension ⊆
  [FP.Structural.Algebra,Math.Value,RegionName,Region,TileDType.int,Spec.Real,JITIO.logicalDimension,
   LogicalProgramIO.Implements,firstMaximum,JITIO.dimensionRow]
#specNonCircular JITRealIO.mathematical avoiding [JITSource.stage1, JITSource.stage2, FP.Structural.exec]
#specNonCircular JITRealDimensionIO.mathematical avoiding [JITSource.dimension, FP.Structural.exec]
#axiomsClean global
#axiomsClean dimension
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Correct
