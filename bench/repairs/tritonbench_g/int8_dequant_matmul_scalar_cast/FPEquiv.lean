import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.ReuseIndicesIO

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.FPEquiv
open VeriTile Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

/-- Reuse original index tiles instead of computing rm/rn a second time.
Full grouped launch, all split/tail/bias branches, integer-dot topology,
FP32 scale/divisor operations, FP16 output and independent per-address atomic
orders remain. The transformation changes only discrete register work. -/
specification reuse_indices_equiv (profile : FixedISAProfile) (s : Shape) (c : Config)
    (admitted : Source.Admitted c) (selected : Selection s c)
    (hasBias : Bool) (schedule : Schedule s c) :
    ReuseIO.original (Quote.nativeInteger profile s c selected) s c hasBias Source.runtimeDivisor schedule ≡[[]]
      ReuseIO.optimized (Quote.nativeInteger profile s c selected) s c hasBias Source.runtimeDivisor schedule :=
  PairIO.Implementation.of_equivalentOnCells
    (ReuseIO.variant (Quote.nativeInteger profile s c selected) s c hasBias Source.runtimeDivisor schedule)
    (ReuseIO.equivalent (Quote.nativeInteger profile s c selected) s c hasBias Source.runtimeDivisor
      schedule selected rfl)

-- Pure index-register elimination uses no floating rewrite or experimental rule.
#axiomsClean reuse_indices_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.FPEquiv
