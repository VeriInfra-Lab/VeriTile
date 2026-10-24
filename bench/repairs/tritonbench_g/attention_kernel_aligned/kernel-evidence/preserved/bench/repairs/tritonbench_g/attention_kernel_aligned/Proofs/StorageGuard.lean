import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.Physical
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Whole
/-- Exact half-open byte-interval overlap predicate used by the wrapper.
Different base addresses alone do not imply this predicate is false. -/
def overlaps (baseA sizeA baseB sizeB : Nat) : Prop :=
  baseA<baseB+sizeB ∧ baseB<baseA+sizeA

theorem interval_apart (baseA sizeA baseB sizeB a b : Nat)
    (guard : ¬overlaps baseA sizeA baseB sizeB)
    (ha : a<sizeA) (hb : b<sizeB) : baseA+a≠baseB+b := by
  unfold overlaps at guard
  omega

theorem half_storage_apart (baseA countA baseB countB a b : Nat)
    (guard : ¬overlaps baseA (2*countA) baseB (2*countB))
    (ha : a<2*countA) (hb : b<2*countB) : baseA+a≠baseB+b :=
  interval_apart baseA (2*countA) baseB (2*countB) a b guard ha hb

#axiomsClean interval_apart
#axiomsClean half_storage_apart
#specNonCircular overlaps avoiding [VeriTile.Triton.FP.Structural.exec]
end VeriTile.Bench.Repairs.AttentionKernelAligned.Whole
