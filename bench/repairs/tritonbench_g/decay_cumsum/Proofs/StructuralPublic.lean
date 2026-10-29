import bench.repairs.tritonbench_g.decay_cumsum.Proofs.FPLogical
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.HostPreparation
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.NativeABI
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.WholeHost

/-! Internal structural FP composition, independent of the Real entry.

The selected profile is FP32. The DSL's `.real` data tag carries arbitrary FP
payloads here, not exact real arithmetic. Direct FP32 loads retain fp32Load;
explicit gate promotion and every store retain their cast callbacks. Prefix
and reverse accumulations retain their original separate multiply/add order.
Prepare's last gate is a direct load; backward's last gate is explicitly cast.
The dg recurrence uses uncast local dq/dk, not rounded stored values.

The independent logical reference contains only these opaque scalar primitives
and recurrences. No primitive identities, reassociation, FMA substitution or
native-FP=Real assertion is used. The public native entry instantiates this
carrier with source-specific FMA graphs and raw FP32 input words; the generic
split syntax here is not its final numerical reference.

The wrapper result includes None, original target identities, all ordered
output bytes and the complete caller byte frame. Ordinary preparation/setup/
reshape/copy failures remain explicit outcomes; no source-success oracle is
assumed. The current README records the remaining local numerical obligations.
-/
namespace VeriTile.Bench.Repairs.DecayCumsum.FPEquiv
open VeriTile.Triton FP.Structural PackedStorage
noncomputable section
variable {α β : Type} [Inhabited α]

def defaultPrecision : ComputeDType := .fp32
abbrev reference := @FPLogical.reference
abbrev source_equiv := @FPStorage.complete
abbrev grid_equiv := @FPLogical.grid_correct

theorem wrapper_equiv (M : Algebra α) (h : Call) (kind : Kind) (storage : Placement h kind) (scale : ℝ)
    (api : Host.API β (Cell α)) (tensors : RegionName → Host.Target)
    (decode : Bool → (Fin 4 → β) → Cell α) (encode : Bool → Cell α → Fin 4 → β)
    (before : Copyback.Heap β) (live : Nat → Prop)
    (ordinaryPreparation : Host.PreparationLaws api h kind storage tensors decode before live)
    (ordinaryCopy : Host.CopyLaws api) :
    Host.Outcome h kind tensors encode
      (fun r i => reference M h kind scale (Host.logicalData h tensors decode before) r (Logical.coordinates h i)) before
      (Host.run api h kind tensors encode
        (Run.run (FPStorage.execute M h kind storage scale) (Grid.schedule h))
        (LogicalStorage.crop h kind storage) before) := by
  apply Host.complete api ordinaryCopy
  intro packed prepared
  obtain ⟨final,run,outputs,_⟩ := FPLogical.grid_correct M h kind storage scale packed
    (Host.logicalData h tensors decode before) (ordinaryPreparation.snapshots packed prepared)
  exact ⟨final,run,outputs⟩

/-- Whole-domain entry: failed preparation carries no allocation package, so
there is no capacity/freshness premise on the failure branch. -/
theorem whole_wrapper_equiv (M : Algebra α) (h : Call) (kind : Kind) (scale : ℝ)
    (api : WholeHost.API h kind β (Cell α)) (tensors : RegionName → Host.Target)
    (decode : Bool → (Fin 4 → β) → Cell α) (encode : Bool → Cell α → Fin 4 → β)
    (before : Copyback.Heap β) (live : Nat → Prop)
    (ordinaryPreparation : WholeHost.PreparationLaws h kind api tensors decode before live)
    (ordinaryCopy : WholeHost.CopyLaws api) :
    Host.Outcome h kind tensors encode
      (fun r i => reference M h kind scale (Host.logicalData h tensors decode before) r (Logical.coordinates h i)) before
      (WholeHost.run h kind api tensors encode
        (fun p => Run.run (FPStorage.execute M h kind p.storage scale) (Grid.schedule h))
        (fun p => LogicalStorage.crop h kind p.storage) before) := by
  apply WholeHost.complete h kind api ordinaryCopy
  intro p prepared
  obtain ⟨final,run,outputs,_⟩ := FPLogical.grid_correct M h kind p.storage scale p.heap
    (Host.logicalData h tensors decode before) (ordinaryPreparation.snapshots p prepared)
  exact ⟨final,run,outputs⟩

/-- Direct public preparation/allocation failure, independent of Placement,
input snapshots, numerical callbacks and any source-success assumption. -/
theorem preparation_failure (M : Algebra α) (h : Call) (kind : Kind) (scale : ℝ)
    (api : WholeHost.API h kind β (Cell α)) (tensors : RegionName → Host.Target)
    (encode : Bool → Cell α → Fin 4 → β) (before : Copyback.Heap β)
    (message : String) (failed : api.prepare=.error message) :
    WholeHost.run h kind api tensors encode
      (fun p => Run.run (FPStorage.execute M h kind p.storage scale) (Grid.schedule h))
      (fun p => LogicalStorage.crop h kind p.storage) before=
      if Logical.cells h=0 then .ok ⟨(),tensors,before⟩ else .error ⟨.preparation,message,before⟩ :=
  WholeHost.preparation_failure h kind api tensors encode _ _ before message failed

#axiomsClean whole_wrapper_equiv
#axiomsClean preparation_failure
#print axioms whole_wrapper_equiv

abbrev empty_return := @Host.empty_return
abbrev validation_failure := @Host.validation_failure
abbrev negative_dimension := Host.negative_dimension
abbrev invalid_tile := Host.invalid_tile
abbrev physical_addresses := NativeABI.source_addresses
abbrev loop_bounds := NativeIndex.loop_bound
abbrev no_overlap_readback := @Copyback.view_readback
abbrev exact_last_write := @Copyback.exact_last_write

#axiomsClean wrapper_equiv
#print axioms wrapper_equiv
-- StatementAudit's #print_fp_assumptions accepts equivalence-calculus heads,
-- not this complete Host.Outcome proposition. The checked axiom audit above
-- covers the actual theorem; no numerical atomic law is a theorem premise.
end
end VeriTile.Bench.Repairs.DecayCumsum.FPEquiv
