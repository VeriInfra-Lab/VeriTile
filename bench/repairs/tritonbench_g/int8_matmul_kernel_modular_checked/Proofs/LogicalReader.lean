import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.IO
import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.IO
import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.GridIO

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.LogicalReader
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
open TileModel
open scoped ProgramIO BigOperators
noncomputable section

/-- Dense signed integer arithmetic: truncate the defined FP32 operands, unpack
four two-bit weight fields, then project to the signed low 32 bits. -/
def denseFormula (layout : Layout) (s : Shape) (xs : PublicIO.Inputs layout s) :
    PublicIO.Outputs layout s := fun port => match port with
  | ⟨0,_⟩ => fun i => (BitVec.ofInt 32
      (∑ f ∈ Finset.range 4, ∑ q ∈ Finset.range (s.k/4),
        ((truncateFloat (PublicIO.inputRow layout s xs i.1 (f*(s.k/4)+q))).getD 0) *
          packedWeight (packedByte (PublicIO.weightRow layout s xs i.2.1 q)) f)).toInt


/-- Complete production grid for any of the fifteen retained configurations:
the signed INT32 output is the low 32 bits of the independent dense integer
sum. IO.lean exposes source precision, strides, channels and exact cast domain. -/
theorem kernel_low32 (cast : Word32 → Byte)
    (castLeaf : ∀ word,DefinedCast word → cast word=referenceCast word)
    (rest : Algebra Word32) (layout : Layout) (s : Shape) (c : Fin 15) (legal : s.legal c)
    (hout : OutputLayout layout s) (separate : layout.output≠layout.input ∧ layout.output≠layout.weight) :
    Spec.Real (PublicIO.matmulIO cast rest layout s c ⊨{PublicIO.Domain layout s} denseFormula layout s) :=
  PublicIO.implements cast castLeaf rest layout s c legal hout separate

end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.LogicalReader

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.LogicalReader
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001 TileModel
open scoped BigOperators
open scoped LogicalProgramIO
noncomputable section

def logicalFormula (s : Shape) (xs : PublicIO.LogicalInputs s) (row : Fin s.m) (col : Fin s.n) : Int :=
  (BitVec.ofInt 32 (∑ f ∈ Finset.range 4, ∑ q ∈ Finset.range (s.k/4),
    ((truncateFloat (PublicIO.inputAt xs row (f*(s.k/4)+q))).getD 0) *
      packedWeight (packedByte (PublicIO.packedAt xs col q)) f)).toInt

theorem full_judgment_iff (cast : Word32 → Byte) (rest : Algebra Word32) (layout : Layout) (s : Shape) (c : Fin 15) :
    (PublicIO.logicalIO cast rest layout s c ⊨{PublicIO.logicalDomain layout s} logicalFormula s) ↔
      (PublicIO.matmulIO cast rest layout s c).ImplementsOn (PublicIO.Domain layout s) (denseFormula layout s) := by
  rfl

end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.LogicalReader
