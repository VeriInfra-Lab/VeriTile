import bench.repairs.tritonbench_g.log_softmax.NativeIO
import bench.repairs.tritonbench_g.log_softmax.Proofs.NativePublicIO
import bench.repairs.tritonbench_g.log_softmax.Proofs.IndexEquivalence

/-! Auxiliary native grid equivalences for forward and backward. The source index-commutation equivalences are in FPEquiv. -/
namespace VeriTile.Bench.Repairs.LogSoftmax.FPEquiv
open VeriTile.Triton FP.Structural FP.ProfiledMemory
open scoped VeriTile.Spec
open VeriTile.Bench.Repairs.SoftmaxFlaggems
noncomputable section

/-- Shared IO descriptors retain complete FP32 source graphs and full grids. -/
abbrev nativeIO := NativeIO.original
abbrev nativeCommuted := NativeIO.commuted
abbrev nativeObservation := NativeIO.observation

/-- Native observations agree for every FP32 bit pattern and arbitrary source
terms, under the same selected reduction layout, ISA primitives and fusion
schedule. Source safety, successful complete grids and both exact exterior
frames are part of the shared judgment. -/
theorem forward_native_grid (M N K BM : Nat) (R : Native.Reductions)
    (I : Native.Instructions) (S : Native.Schedule) :
    (nativeIO .forward R M N K BM).ObservedEquivalent
      (nativeCommuted .forward R M N K BM) (nativeObservation .forward R M N K BM I S) :=
  NativePublicIO.observed_equivalence .forward R M N K BM I S

/-- The same unchanged native fusion selection is applied to both backward
source graphs; fused and separately rounded schedules are never equated. -/
theorem backward_native_grid (M N K BM : Nat) (R : Native.Reductions)
    (I : Native.Instructions) (S : Native.Schedule) :
    (nativeIO .backward R M N K BM).ObservedEquivalent
      (nativeCommuted .backward R M N K BM) (nativeObservation .backward R M N K BM I S) :=
  NativePublicIO.observed_equivalence .backward R M N K BM I S

#axiomsClean forward_native_grid

#axiomsClean backward_native_grid

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LogSoftmax.FPEquiv
