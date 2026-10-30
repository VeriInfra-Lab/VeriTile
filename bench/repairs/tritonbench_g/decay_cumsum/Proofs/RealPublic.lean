import bench.repairs.tritonbench_g.decay_cumsum.Proofs.RealLogical
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.HostPreparation
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.NativeABI
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.WholeHost

/-! Independent Real composition helper for the three repaired wrappers.

`reference` is the ordinary mathematical target: forward chunk-prefix sums;
prepare q*exp2(g)*scale and k*exp2(last-g); backward updated dq/dk and a
chunk-local reverse sum of dq*q-dk*k. Every read role uses its entry snapshot.
The source is the shared FP32 source's Real projection, proved by reusing the
unchanged original complete per-program executions and their memory frames.

Both main and short-tail grids, every logical coordinate, and deterministic
ordered byte writeback are included. This generic helper supports broader
target descriptors; the checked public entry binds exact contiguous shapes
and absent negative flags. Cross-role aliases retain their entry snapshots and
source-order last physical byte writes.

Ordinary input-copy/allocation and copy_ contracts are explicit. Preparation,
setup and copy failures remain outcomes. The generic output-shape check is
vacuous under the public shape guard. No successful source
execution or output-value premise is assumed. Native launch-grid compatibility
and native FP-to-Real numerical relations are separately deferred.
-/
namespace VeriTile.Bench.Repairs.DecayCumsum.Correct
open VeriTile.Triton PackedStorage
noncomputable section
variable {β : Type}

abbrev reference := RealLogical.reference
abbrev source_correct := RealStorage.complete
abbrev grid_correct := RealLogical.grid_correct

theorem wrapper_correct (h : Call) (kind : Kind) (storage : Placement h kind) (scale : ℝ)
    (api : Host.API β ℝ) (tensors : RegionName → Host.Target)
    (decode : Bool → (Fin 4 → β) → ℝ) (encode : Bool → ℝ → Fin 4 → β)
    (before : Copyback.Heap β) (live : Nat → Prop)
    (ordinaryPreparation : Host.PreparationLaws api h kind storage tensors decode before live)
    (ordinaryCopy : Host.CopyLaws api) :
    Host.Outcome h kind tensors encode
      (fun r i => reference h kind scale (Host.logicalData h tensors decode before) r (Logical.coordinates h i)) before
      (Host.run api h kind tensors encode
        (Run.run (RealStorage.execute h kind storage scale) (Grid.schedule h))
        (LogicalStorage.crop h kind storage) before) := by
  apply Host.complete api ordinaryCopy
  intro packed prepared
  obtain ⟨final,run,outputs,_⟩ := RealLogical.grid_correct h kind storage scale packed
    (Host.logicalData h tensors decode before) (ordinaryPreparation.snapshots packed prepared)
  exact ⟨final,run,outputs⟩

/-- Whole-domain entry: actual allocation metadata is required only when
preparation returns a successful package. Failed allocations need no Placement. -/
theorem whole_wrapper_correct (h : Call) (kind : Kind) (scale : ℝ)
    (api : WholeHost.API h kind β ℝ) (tensors : RegionName → Host.Target)
    (decode : Bool → (Fin 4 → β) → ℝ) (encode : Bool → ℝ → Fin 4 → β)
    (before : Copyback.Heap β) (live : Nat → Prop)
    (ordinaryPreparation : WholeHost.PreparationLaws h kind api tensors decode before live)
    (ordinaryCopy : WholeHost.CopyLaws api) :
    Host.Outcome h kind tensors encode
      (fun r i => reference h kind scale (Host.logicalData h tensors decode before) r (Logical.coordinates h i)) before
      (WholeHost.run h kind api tensors encode
        (fun p => Run.run (RealStorage.execute h kind p.storage scale) (Grid.schedule h))
        (fun p => LogicalStorage.crop h kind p.storage) before) := by
  apply WholeHost.complete h kind api ordinaryCopy
  intro p prepared
  obtain ⟨final,run,outputs,_⟩ := RealLogical.grid_correct h kind p.storage scale p.heap
    (Host.logicalData h tensors decode before) (ordinaryPreparation.snapshots p prepared)
  exact ⟨final,run,outputs⟩

/-- Direct allocation/input-copy preparation-failure behavior,
with neither successful allocation metadata nor ordinary success laws. -/
theorem preparation_failure (h : Call) (kind : Kind) (scale : ℝ)
    (api : WholeHost.API h kind β ℝ) (tensors : RegionName → Host.Target)
    (encode : Bool → ℝ → Fin 4 → β) (before : Copyback.Heap β)
    (message : String) (failed : api.prepare=.error message) :
    WholeHost.run h kind api tensors encode
      (fun p => Run.run (RealStorage.execute h kind p.storage scale) (Grid.schedule h))
      (fun p => LogicalStorage.crop h kind p.storage) before=
      if Logical.cells h=0 then .ok ⟨(),tensors,before⟩ else .error ⟨.preparation,message,before⟩ :=
  WholeHost.preparation_failure h kind api tensors encode _ _ before message failed

#axiomsClean whole_wrapper_correct
#axiomsClean preparation_failure
#print axioms whole_wrapper_correct

abbrev empty_return := @Host.empty_return
abbrev validation_failure := @Host.validation_failure
abbrev negative_dimension := Host.negative_dimension
abbrev invalid_tile := Host.invalid_tile
abbrev physical_addresses := NativeABI.source_addresses
abbrev loop_bounds := NativeIndex.loop_bound
abbrev no_overlap_readback := @Copyback.view_readback
abbrev exact_last_write := @Copyback.exact_last_write

#axiomsClean wrapper_correct
#print axioms wrapper_correct
end
end VeriTile.Bench.Repairs.DecayCumsum.Correct
