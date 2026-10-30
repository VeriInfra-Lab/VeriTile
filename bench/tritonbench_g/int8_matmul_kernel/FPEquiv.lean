import bench.tritonbench_g.int8_matmul_kernel.Proofs.ConfigurationEquivalence

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.FPEquiv
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
open TileModel
open scoped VeriTile.Spec
noncomputable section

/-- Compare two actual retained autotune programs. Matrix tiles, source
specializations and complete grids may differ. The packed input/cast domain
and signed low32 result are unchanged; no no-overflow restriction is added.
The defective carry-fed predecessor is not used as a comparison program. -/
specification configuration_equiv (cast : Word32 → Byte)
    (castLeaf : ∀ word,DefinedCast word → cast word=referenceCast word)
    (rest : Algebra Word32) (layout : Layout) (s : Shape) (left right : Fin 15)
    (hl : s.legal left) (hr : s.legal right) (hout : OutputLayout layout s)
    (separate : layout.output≠layout.input ∧ layout.output≠layout.weight) :
    PublicIO.original cast rest layout s left ≡[[]]
      PublicIO.configured cast rest layout s left right :=
  ProgramIO.Implementation.of_equivalentOn (PublicIO.configuredVariant cast rest layout s left right)
    (PublicIO.configurations_equivalent cast castLeaf rest layout s left right hl hr hout separate)

#print_fp_assumptions configuration_equiv
#axiomsClean configuration_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.FPEquiv
