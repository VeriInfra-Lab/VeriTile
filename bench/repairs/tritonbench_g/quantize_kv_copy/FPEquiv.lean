import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.Pair

namespace VeriTile.Bench.Repairs.QuantizeKvCopy.FPEquiv
open VeriTile.Triton FP.Structural Kernel KernelModels
open scoped VeriTile.Spec
noncomputable section
local instance instPairWord : Inhabited KernelModels.Word := ⟨KernelModels.zero⟩

/-- The two complete grouped KV quantization programs differ only by scheduling
pure output address formation before the FP32 quantization pipeline. No floating
operation is changed. The original metadata, all-word domain, two output channels,
full token/head grid, source safety and exterior frames are retained. -/
specification address_early_equiv (instructions : NativeInstructions) (tree : ReductionTree)
    (p : Parameters) (n : Nat) (dest : Nat → Nat) (covers : p.groups ≤ p.block)
    (unique : ∀ a b, a<n → b<n →
      KernelGridResult.target p dest a = KernelGridResult.target p dest b → a=b) :
    Pair.original instructions tree p n dest ≡[[]] Pair.addressEarly instructions tree p n dest :=
  ProgramIO.Implementation.of_equivalentOn
    (Pair.variant (KernelModels.nativeOps instructions tree) p n dest)
    (Pair.equivalent instructions tree p n dest covers unique)

#axiomsClean address_early_equiv
end
end VeriTile.Bench.Repairs.QuantizeKvCopy.FPEquiv
