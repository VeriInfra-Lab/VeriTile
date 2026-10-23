import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.PairEquivalence
namespace VeriTile.Bench.Repairs.LayerNormWelfold.FPEquiv
open VeriTile.Triton FP.Structural Reuse
open scoped VeriTile.Spec
noncomputable section
abbrev original := @PairIO.original
abbrev rowOffsetReused := @PairIO.rowOffsetReused

/-- The complete two-pass LayerNorm row grid reuses D*xrow for the X load
and BF16 Y store in each normalization chunk. All FP32 casts, ordered sums,
subtractions, multiplications, additions, rsqrt and BF16 stores are retained.
The original arbitrary-word input domain, mean/Rstd outputs, access safety,
success and both exterior frames are part of this program-pair judgment. -/
specification row_offset_reused_equiv {α : Type} [Inhabited α]
    (M : Algebra α) (Y Mean Rstd : RegionName) (S D : Nat)
    (c : Configuration) (sep : Separation "X" "W" "Bias" Y Mean Rstd) :
    original M Y Mean Rstd S D c.block c.tree ≡[[]]
      rowOffsetReused M Y Mean Rstd S D c.block c.tree :=
  PairEquivalence.equivalent M Y Mean Rstd S D c sep
#axiomsClean row_offset_reused_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormWelfold.FPEquiv
