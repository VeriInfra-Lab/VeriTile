import bench.repairs.tritonbench_g.max_reduction.Proofs.SourceEquivalence

namespace VeriTile.Bench.Repairs.MaxReduction.FPEquiv
open VeriTile.Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

/-- Actual complete FP32 sources and their existing integer-only variants.
The same reduction engine fixes the value/index tree on both sides. -/
abbrev stage1Original (e : Source.Engine A) := RawFP.stage1Program e Bool.false
abbrev stage1IndexCommuted (e : Source.Engine A) := RawFP.stage1Program e Bool.true
abbrev stage2Original (e : Source.Engine A) := RawFP.stage2Program e Bool.false
abbrev stage2MaskComplement (e : Source.Engine A) := RawFP.stage2Program e Bool.true
abbrev dimensionOriginal (e : Source.Engine A) := RawFP.dimensionProgram e Bool.false
abbrev dimensionIndexCommuted (e : Source.Engine A) := RawFP.dimensionProgram e Bool.true
def assumptions := RawFP.assumptions

specification stage1_index_equiv [Inhabited A] (e : Source.Engine A)
    (inp out : RegionName) (N B : Nat) (hB : 0<B) :
    stage1Original e inp out N B ≡[assumptions]
      stage1IndexCommuted e inp out N B :=
  RawFP.stage1_index_equiv e inp out N B hB

specification stage2_mask_equiv [Inhabited A] (e : Source.Engine A)
    (inp out : RegionName) (N B : Nat) (hB : 0<B) :
    stage2Original e inp out N B ≡[assumptions]
      stage2MaskComplement e inp out N B :=
  RawFP.stage2_mask_equiv e inp out N B hB

/-- Both value and signed-index stores agree for every program ID, including
zero K and aliased output regions. No floating rewrite is used. -/
specification dimension_index_equiv [Inhabited A] (e : Source.Engine A)
    (inp ov oi : RegionName) (M N K BM BN : Nat) (hBN : 0<BN) :
    dimensionOriginal e inp ov oi M N K BM BN ≡[assumptions]
      dimensionIndexCommuted e inp ov oi M N K BM BN :=
  RawFP.dimension_index_equiv e inp ov oi M N K BM BN hBN

#print_fp_assumptions stage1_index_equiv
#print_fp_assumptions stage2_mask_equiv
#print_fp_assumptions dimension_index_equiv
#axiomsClean stage1_index_equiv
#axiomsClean stage2_mask_equiv
#axiomsClean dimension_index_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MaxReduction.FPEquiv
