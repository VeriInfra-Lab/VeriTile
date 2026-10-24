import bench.tritonbench_g.max_reduction.Proofs.StageIO
import bench.tritonbench_g.max_reduction.Proofs.DimensionIO

/-! Retained public Real stage and dimension contracts. Stage proofs live in
Proofs.StageIO; declaration names, types and source domains remain compatible. -/
namespace VeriTile.Bench.TritonBenchG.MaxReduction
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedTileKernelIO₁

/-- Full second-stage IO correctness against a finite mathematical maximum. -/
specification stage2_math_correct (mid out : RegionName) (M B : Nat) (hB : 0 < B) :
    stage2IO mid out M B ⊨ fun _ xs _ => blockMaximum M B 0 xs :=
  StageIOProof.stage2_math_correct mid out M B hB

/-- Each source version independently implements the finite maximum of the
masked original input window, terminates, and preserves every other flat cell. -/
specification stage1_real_correct (storage : Storage) (v : Bool)
    (inp out : RegionName) (M B : Nat) (hB : 0 < B) :
    VeriTile.Spec.Real (typedStage1IO storage v inp out M B ⊨
      fun pid xs _ => blockMaximum M B pid xs) :=
  StageIOProof.stage1_real_correct storage v inp out M B hB

/-- Each second-stage source independently computes the finite maximum of the
masked intermediate window, with successful execution and a full flat frame. -/
specification stage2_real_correct (storage : Storage) (v : Bool)
    (inp out : RegionName) (M B : Nat) (hB : 0 < B) :
    VeriTile.Spec.Real (typedStage2IO storage v inp out M B ⊨
      fun _ xs _ => blockMaximum M B 0 xs) :=
  StageIOProof.stage2_real_correct storage v inp out M B hB

/-- Both full indexed-reduction sources independently compute row maxima and
leftmost indices from the complete masked input window. The signed index
output, value output, successful execution and full flat frame are all covered.
Distinct output regions and injective row addresses prevent conflicting stores. -/
specification dimension_real_correct (storage : Storage) (v : Bool)
    (inp ov oi : RegionName) (M N K BM BN : Nat)
    (hN : 0 < BN) (hRegions : ov ≠ oi)
    (hOutInj : Function.Injective (fun i : Fin BM => i.val * K)) :
    VeriTile.Spec.Real (DimensionIO storage v inp ov oi M N K BM BN) := by
  exact dimensionIO_correct storage v inp ov oi M N K BM BN hN hRegions hOutInj

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.MaxReduction
