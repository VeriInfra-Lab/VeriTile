import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceReadback

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.SourceSuffix
open VeriTile.Triton FP.Structural
noncomputable section

def readOutput {α : Type} [Inhabited α] (abi : SourceABI) (p : KernelArgs)
    (s : State α) (r c : Nat) : α :=
  toFloat (storageType abi.output)
    ((s.mem p.output (r*p.outputRowStride+c*p.outputColumnStride)).read (storageType abi.output).toTileDType)

def OutputMemory {α : Type} [Inhabited α] (abi : SourceABI) (p : KernelArgs)
    (shape : Shape) (s : State α) (previous : Nat → Nat → α) : Prop :=
  ∀ r c, r < shape.outputRows → c < shape.outputColumns → readOutput abi p s r c = previous r c

theorem outputValues_active {α : Type} [Inhabited α] (M : Algebra α) (abi : SourceABI)
    (p : KernelArgs) (c : Coordinates) (s : State α) (acc : Values α .real [32,32])
    (i : TileIndex [32,32]) (hi : active p c i) :
    outputValues M abi p c s acc i = finish M abi (acc i)
      (if abi.output = .single then M.fp32Load (readOutput abi p s (row c i) (column p c i))
        else readOutput abi p s (row c i) (column p c i)) := by
  simp only [outputValues,if_pos hi,active_offset p c i hi,readOutput]

#axiomsClean outputValues_active
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.SourceSuffix
