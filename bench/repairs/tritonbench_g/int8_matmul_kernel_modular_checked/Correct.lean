import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001 TileModel
open scoped BigOperators
open scoped LogicalProgramIO
noncomputable section

/-- Signed low32 dense integer sum, with the original defined FP32 cast and packed fields. -/
def denseFormula (s : Shape) (xs : PublicIO.LogicalInputs s) (row : Fin s.m) (col : Fin s.n) : Int :=
  (BitVec.ofInt 32 (∑ f ∈ Finset.range 4, ∑ q ∈ Finset.range (s.k/4),
    ((truncateFloat (PublicIO.inputAt xs row (f*(s.k/4)+q))).getD 0) *
      packedWeight (packedByte (PublicIO.packedAt xs col q)) f)).toInt


/-- Complete production grid for any retained legal configuration; overflow and exact cast domains remain. -/
specification kernel_low32 (cast : Word32 → Byte) (castLeaf : ∀ word,DefinedCast word → cast word=referenceCast word)
    (rest : Algebra Word32) (layout : Layout) (s : Shape) (c : Fin 15) (legal : s.legal c)
    (hout : OutputLayout layout s) (separate : layout.output≠layout.input ∧ layout.output≠layout.weight) :
    Spec.Real (PublicIO.logicalIO cast rest layout s c ⊨{PublicIO.logicalDomain layout s} denseFormula s) :=
  (LogicalReader.full_judgment_iff cast rest layout s c).mpr
    (LogicalReader.kernel_low32 cast castLeaf rest layout s c legal hout separate)
#axiomsClean kernel_low32

end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
