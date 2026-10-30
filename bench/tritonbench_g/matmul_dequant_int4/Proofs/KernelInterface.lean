import bench.tritonbench_g.matmul_dequant_int4.Proofs.Grid

namespace VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.KernelInterface
open VeriTile.Triton FP.Structural
noncomputable section

/-- Independent raw packed-input nibble difference at a logical coordinate. -/
def difference {α : Type} [Inhabited α] (p : Source.Parameters) (s : State α)
    (x : KernelGrid.Coordinate p) : Int :=
  Source.nibble ((s.mem p.weight (x.1.val/8*p.weightRowStride+x.2.val*p.weightColumnStride)).read .int) x.1.val -
    Source.nibble ((s.mem p.zero (x.1.val/p.groupSize*p.zeroRowStride+x.2.val/8*p.zeroColumnStride)).read .int) x.2.val

def scale {α : Type} [Inhabited α] (f : Source.Profile) (p : Source.Parameters)
    (s : State α) (x : KernelGrid.Coordinate p) : α :=
  match f with
  | .half => (s.mem p.scale (x.1.val/p.groupSize*p.scaleRowStride+x.2.val*p.scaleColumnStride)).read .fp16
  | .single => (s.mem p.scale (x.1.val/p.groupSize*p.scaleRowStride+x.2.val*p.scaleColumnStride)).read .fp32

theorem difference_bounds {α : Type} [Inhabited α] (p : Source.Parameters)
    (s : State α) (x : KernelGrid.Coordinate p) :
    -15≤difference p s x ∧ difference p s x≤15 := by
  have a := VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPBits.mask_bound
    (BitVec.ofInt 32 (((s.mem p.weight (x.1.val/8*p.weightRowStride+x.2.val*p.weightColumnStride)).read .int) >>> (x.1.val%8*4)))
  have b := VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPBits.mask_bound
    (BitVec.ofInt 32 (((s.mem p.zero (x.1.val/p.groupSize*p.zeroRowStride+x.2.val/8*p.zeroColumnStride)).read .int) >>> (x.2.val%8*4)))
  change 0≤Source.nibble _ x.1.val ∧ Source.nibble _ x.1.val≤15 at a
  change 0≤Source.nibble _ x.2.val ∧ Source.nibble _ x.2.val≤15 at b
  unfold difference
  omega

#axiomsClean difference_bounds
end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.KernelInterface
