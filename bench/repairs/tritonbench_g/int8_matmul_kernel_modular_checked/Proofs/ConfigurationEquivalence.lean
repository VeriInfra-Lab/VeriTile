import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.GridIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.PublicIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
open TileModel
noncomputable section

theorem configured_io (cast : Word32 → Byte) (rest : Algebra Word32) (layout : Layout)
    (s : Shape) (left right : Fin 15) :
    (configuredVariant cast rest layout s left right).toProgramIO = matmulIO cast rest layout s right := rfl

/-- The independent dense low32 formula is invariant under every admitted
configuration. Safety, complete-grid success and exterior frames are proved
for both actual schedules, including zero M/N and zero K. -/
theorem configurations_equivalent (cast : Word32 → Byte)
    (castLeaf : ∀ word,DefinedCast word → cast word=referenceCast word)
    (rest : Algebra Word32) (layout : Layout) (s : Shape) (left right : Fin 15)
    (hl : s.legal left) (hr : s.legal right) (hout : OutputLayout layout s)
    (separate : layout.output≠layout.input ∧ layout.output≠layout.weight) :
    (matmulIO cast rest layout s left).EquivalentOn
      (configuredVariant cast rest layout s left right) (Domain layout s) := by
  apply ProgramIO.EquivalentOn.of_implementsOn
    (implements cast castLeaf rest layout s left hl hout separate)
  change (matmulIO cast rest layout s right).ImplementsOn (Domain layout s) (formula layout s)
  exact implements cast castLeaf rest layout s right hr hout separate

#axiomsClean configurations_equivalent
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.PublicIO
