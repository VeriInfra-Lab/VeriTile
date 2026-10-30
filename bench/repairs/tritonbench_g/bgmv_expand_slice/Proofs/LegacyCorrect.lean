/- Historical padded/snapshot wrapper theorem, retained as prior proof credit. -/
import bench.repairs.tritonbench_g.bgmv_expand_slice.Kernels
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.RealHost
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.CallerViews
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.HostCoordinates
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.NativeControl

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.Correct
open VeriTile.Triton Snapshot ReturnedStorage
noncomputable section

abbrev caller (views : CallerViews.Views) (memory : CallerViews.Memory ℝ) :=
  CallerViews.logical LogicalReturn.realCodec views memory

def scratch (h : Call) (state : BlockState) (batch row : Nat) : ℝ :=
  state.readMem "Y" (Geometry.outputOffset h batch row 0)

/-- Complete source grid, independent caller-coordinate K-term GEMV, and
logical readback into the original supplied output descriptor. Input aliases,
strides and negative views are resolved at the caller-entry snapshot boundary.
Only ordinary local snapshot/copy and admitted-grid contracts are parameters. -/
theorem correct (h : Call) (schedule : Schedule h) (views : CallerViews.Views)
    (memory : CallerViews.Memory ℝ) (packed : Packed ℝ h (caller views memory) 0)
    (allocations : CallerViews.Allocations views) (viewFacts : CallerViews.Valid h views allocations)
    (guard : ViewGuard.Admitted views.output)
    (before : LogicalReturn.Words ℝ)
    (initialOutput : ∀ i : Coord views.output,
      LogicalReturn.read LogicalReturn.realCodec views.output before i=(caller views memory).output i.1.val i.2.val)
    (valid : ∀ b,b<h.batches → (caller views memory).indices b=-1 ∨
      (0 ≤ (caller views memory).indices b ∧ ((caller views memory).indices b).toNat<h.adapters)) :
    ∃ state,RealGrid.run h schedule.programs (RealHost.initial h (caller views memory) packed)=some state ∧
      (∀ i : Coord views.output,∀ byte : Fin 4,views.output.address i byte<2^64) ∧
      (∀ i : Coord views.output,Active h views.output (caller views memory).indices i →
        LogicalReturn.read LogicalReturn.realCodec views.output
          (LogicalReturn.copy LogicalReturn.realCodec h views.output (caller views memory).indices (scratch h state) before) i=
            RealHost.math h (caller views memory) i.1.val (i.2.val-h.slice) ((caller views memory).indices i.1.val).toNat) ∧
      (∀ i : Coord views.output,¬Active h views.output (caller views memory).indices i →
        LogicalReturn.read LogicalReturn.realCodec views.output
          (LogicalReturn.copy LogicalReturn.realCodec h views.output (caller views memory).indices (scratch h state) before) i=
            (caller views memory).output i.1.val i.2.val) ∧
      (∀ owner address,(¬∃ i : Coord views.output,Active h views.output (caller views memory).indices i ∧
          owner=views.output.owner ∧ address=views.output.address i ⟨0,by decide⟩) →
        LogicalReturn.copy LogicalReturn.realCodec h views.output (caller views memory).indices (scratch h state) before owner address=
          before owner address) := by
  obtain ⟨state,run,values,_⟩ := RealHost.complete h schedule (caller views memory) packed valid
  refine ⟨state,run,?_,?_,?_,?_⟩
  · intro i byte
    exact (ReturnedStorage.backed_byte views.output viewFacts.output i byte).2.2
  · intro i active
    rw [LogicalReturn.readback _ _ _ (ViewGuard.writable views.output guard) _ _ _ i active]
    exact values (HostCoordinates.program h views.output (caller views memory).indices i active) active.2.2.2
  · intro i inactive
    rw [LogicalReturn.inactive_read _ _ _ (ViewGuard.writable views.output guard) _ _ _ i inactive]
    exact initialOutput i
  · intro owner address outside
    exact LogicalReturn.frame _ _ _ _ _ _ owner address outside

/-- The original caller output values in the independent formula are tied to
that same supplied descriptor, including its lazy negative conversion. -/
theorem caller_output (views : CallerViews.Views) (memory : CallerViews.Memory ℝ)
    (i : Coord views.output) : (caller views memory).output i.1.val i.2.val=
      LogicalReturn.realCodec.decode views.output.negative
        (memory.fp32 (views.output.address i ⟨0,by decide⟩)) :=
  CallerViews.supplied_output LogicalReturn.realCodec views memory i

#axiomsClean correct
#axiomsClean caller_output
#axiomsClean NativeIndex.source_addresses
#axiomsClean NativeControl.metadata_address
#axiomsClean NativeControl.cdiv_word
end
end VeriTile.Bench.Repairs.BgmvExpandSlice.Correct
