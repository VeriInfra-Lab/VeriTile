import bench.repairs.tritonbench_g.max_reduction.IO
import bench.repairs.tritonbench_g.max_reduction.Proofs.PublicEquivalence

namespace VeriTile.Bench.Repairs.MaxReduction.NativeCorrect
open VeriTile.Triton FP.Structural
noncomputable section

/-- Actual source schedules, arbitrary primitive reduction payloads and all
original region aliases. Only the displayed integer index/mask syntax changes. -/
theorem stage1_index_equiv [Inhabited A] (e : Source.Engine A)
    (inp out : RegionName) (N B pid : Nat) (hB : 0<B) :
    (PublicIO.stageIO e Bool.false inp out N B pid).Equivalent
      (PublicIO.stageCandidate e Bool.false inp out N B pid) :=
  PublicIO.stage_equivalent e Bool.false inp out N B pid hB

theorem stage2_mask_equiv [Inhabited A] (e : Source.Engine A)
    (inp out : RegionName) (N B pid : Nat) (hB : 0<B) :
    (PublicIO.stageIO e Bool.true inp out N B pid).Equivalent
      (PublicIO.stageCandidate e Bool.true inp out N B pid) :=
  PublicIO.stage_equivalent e Bool.true inp out N B pid hB

/-- The same reduction engine and schedule produce identical value/index
outputs and exterior cells. Zero K and coincident output regions remain in the
original equivalence domain; no numerical or special-value restriction is added. -/
theorem dimension_index_equiv [Inhabited A] (e : Source.Engine A)
    (inp ov oi : RegionName) (M N K BM BN p k : Nat) (hBN : 0<BN) :
    (PublicIO.dimensionIO e id id inp ov oi M N K BM BN p k).Equivalent
      (PublicIO.dimensionCandidate e inp ov oi M N K BM BN p k) :=
  PublicIO.dimension_equivalent e inp ov oi M N K BM BN p k hBN

#stmtSurfaceSubset stage1_index_equiv ⊆
  [Source.Engine,RegionName,PublicIO.stageIO,PublicIO.stageCandidate,ProgramIO.Equivalent]
#stmtSurfaceSubset stage2_mask_equiv ⊆
  [Source.Engine,RegionName,PublicIO.stageIO,PublicIO.stageCandidate,ProgramIO.Equivalent]
#stmtSurfaceSubset dimension_index_equiv ⊆
  [Source.Engine,RegionName,PublicIO.dimensionIO,PublicIO.dimensionCandidate,ProgramIO.Equivalent]
-- The reused complete-source transformations have no floating rewrite laws.
#print_fp_assumptions RawFP.stage1_index_equiv
#print_fp_assumptions RawFP.stage2_mask_equiv
#print_fp_assumptions RawFP.dimension_index_equiv
#axiomsClean stage1_index_equiv
#axiomsClean stage2_mask_equiv
#axiomsClean dimension_index_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MaxReduction.NativeCorrect
