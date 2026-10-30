import bench.repairs.tritonbench_g.int8_matmul_kernel.Proofs.Contract
import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.TensorStorage

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
open TileModel TensorModel
noncomputable section

/-- Exactly the existing checked FP32/UINT8 input contract; Python AST
identity checks establish that every guard and all fifteen configs persist. -/
abbrev InputContract := VeriTile.Bench.Repairs.Int8MatmulKernel.InputContract

theorem admitted_defined (request : Request) (aPointer bPointer out : Nat) (before : Memory Word32)
    (admitted : InputContract request aPointer bPointer before) :
    DefinedInputs request out before := by
  intro m hm k hk
  exact admitted.defined m k hm hk

/-- The arbitrary-Real ideal product is a distinct specification. Native
FP32 truncation and INT32 wrapping are not silently identified with it. -/
def idealRealTarget (K : Nat) (xs : Nat → ℝ) (ys : Nat → Byte) : ℝ :=
  ∑ f ∈ Finset.range 4, ∑ q ∈ Finset.range (K/4),
    xs (f*(K/4)+q) * (packedWeight (ys q) f : ℝ)

#axiomsClean admitted_defined
#specNonCircular idealRealTarget avoiding [source, gridMemory, resultMemory, modularAccumulation]
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
