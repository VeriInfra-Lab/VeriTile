import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.GridIO

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.PublicIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
open TileModel
noncomputable section

theorem native_formula_eq (cast : Word32 → Byte)
    (instruction : Instruction → (Fin 32 → Byte) → (Fin 32 → Byte) → Word32 → Word32)
    (add32 : Word32 → Word32 → Word32) (leaves : NumericLeaves cast instruction)
    (addLeaf : Add32Leaf add32) (layout : Layout) (s : Shape) (c : Fin 15) (legal : s.legal c)
    (xs : Inputs layout s) (domain : Domain layout s xs) :
    formula layout s xs=nativeFormula cast instruction add32 layout s c xs := by
  funext port i
  have port0 : port=⟨0,by simp⟩ := by
    apply Fin.ext
    change port.val=0
    have hp : port.val<1 := port.isLt
    omega
  subst port
  change (denseWordTarget s.k (inputRow layout s xs i.1) (weightRow layout s xs i.2.1)).toInt =
    (nativeModularAccumulation cast instruction add32 s c
      (inputRow layout s xs i.1) (weightRow layout s xs i.2.1)).toInt
  symm
  apply congrArg BitVec.toInt
  apply nativeAccumulation_low32 cast instruction add32 leaves addLeaf s c legal
  intro k hk
  simpa [inputRow,hk] using domain.1 (i.1,⟨k,hk⟩,PUnit.unit)

theorem native_implements (cast : Word32 → Byte)
    (instruction : Instruction → (Fin 32 → Byte) → (Fin 32 → Byte) → Word32 → Word32)
    (add32 : Word32 → Word32 → Word32) (leaves : NumericLeaves cast instruction)
    (addLeaf : Add32Leaf add32) (rest : Algebra Word32)
    (layout : Layout) (s : Shape) (c : Fin 15) (legal : s.legal c)
    (hout : OutputLayout layout s) (separate : layout.output≠layout.input ∧ layout.output≠layout.weight) :
    (matmulIO cast rest layout s c).ImplementsOn (Domain layout s)
      (nativeFormula cast instruction add32 layout s c) := by
  intro xs hd bounds hb initial hin
  have result := implements cast leaves.cast rest layout s c legal hout separate xs hd bounds hb initial hin
  rw [native_formula_eq cast instruction add32 leaves addLeaf layout s c legal xs hd] at result
  exact result

#axiomsClean native_formula_eq
#axiomsClean native_implements
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.PublicIO
