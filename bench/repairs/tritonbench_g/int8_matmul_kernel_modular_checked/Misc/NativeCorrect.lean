import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.IO
import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.NativeIO

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
open TileModel
noncomputable section

/-- Same full source grid, observed through the existing FP32-to-signed8,
fixed K32 instruction and two-word add.u32 primitive ports. Their conformance
premises are retained; no kernel transformation or numerical experiment is added. -/
theorem native_kernel (cast : Word32 → Byte)
    (instruction : Instruction → (Fin 32 → Byte) → (Fin 32 → Byte) → Word32 → Word32)
    (add32 : Word32 → Word32 → Word32) (leaves : NumericLeaves cast instruction)
    (addLeaf : Add32Leaf add32) (rest : Algebra Word32)
    (layout : Layout) (s : Shape) (c : Fin 15) (legal : s.legal c)
    (hout : OutputLayout layout s) (separate : layout.output≠layout.input ∧ layout.output≠layout.weight) :
    (PublicIO.matmulIO cast rest layout s c).ImplementsOn (PublicIO.Domain layout s)
      (PublicIO.nativeFormula cast instruction add32 layout s c) :=
  PublicIO.native_implements cast instruction add32 leaves addLeaf rest layout s c legal hout separate

#stmtSurfaceSubset native_kernel ⊆ [Word32,Byte,Instruction,NumericLeaves,Add32Leaf,
  VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001.Word,RegionName,
  FP.Structural.Algebra,Layout,Layout.output,Layout.input,Layout.weight,Shape,Shape.legal,OutputLayout,
  PublicIO.matmulIO,ProgramIO.ImplementsOn,PublicIO.Domain,PublicIO.nativeFormula]
#axiomsClean native_kernel
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
