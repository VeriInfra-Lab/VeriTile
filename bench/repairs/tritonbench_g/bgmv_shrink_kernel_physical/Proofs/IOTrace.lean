import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOLocality
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.CanonicalTrace

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.IOTrace
open VeriTile.Triton FP.Structural FPValues
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Linearization
noncomputable section
set_option maxHeartbeats 1500000
variable {α γ : Type} [Inhabited α]
variable (M : Algebra α) (P : Primitives α) (h : Call)
    (indices : Nat → Int) (scale : α) (orders : Orders h) (observe : α → γ)
local notation "kernelIO" => IO.programIO M P h indices scale orders observe

theorem initial_value (vals : (kernelIO).Inputs) (base : State α)
    (binding : (kernelIO).InputCells vals base) (initial : (kernelIO).RawOutputs)
    (output : (kernelIO).OutputCells vals initial base) (b : Fin h.batches) (n : Fin h.N) :
    initial ⟨0,(b,n,PUnit.unit)⟩ = ofFloat h.outputType
      (FPWorkset.initial h (IOValues.data h indices scale vals) b n) := by
  have input := binding 3 (b,n,PUnit.unit) trivial
  have out := output ⟨0,(b,n,PUnit.unit)⟩ trivial
  change base.mem "Y" (outputOffset h b.val n.val) =
    Cell.mk h.outputType.toTileDType (mapValue id h.outputType.toTileDType (vals 3 (b,n,PUnit.unit))) at input
  rw [IOBinding.map_id] at input
  change base.mem "Y" (outputOffset h b.val n.val) =
    Cell.mk h.outputType.toTileDType (initial ⟨0,(b,n,PUnit.unit)⟩) at out
  have eq := congrArg (Cell.read h.outputType.toTileDType) (out.symm.trans input)
  change initial ⟨0,(b,n,PUnit.unit)⟩ =
    ofFloat h.outputType (IOValues.output h indices vals (outputOffset h b.val n.val))
  rw [IOValues.output_value,IOBinding.float_roundtrip]
  simpa only [Cell.read,eq_self_iff_true,dite_true] using eq

/-- Every transition of the old local linearization is realized by a safe,
successful complete source invocation on the same selected current cell. -/
theorem of_local_trace (vals : (kernelIO).Inputs) (domain : IO.domain h indices vals)
    (base : State α) (binding : (kernelIO).InputCells vals base) (initial : (kernelIO).RawOutputs)
    (b : Fin h.batches) (n : Fin h.N) (active : indices b.val ≠ -1)
    {order : List (Fin 64)} {before after : α}
    (trace : Trace (FPWorkset.step M P h (IOValues.data h indices scale vals) b n) order before after) :
    (kernelIO).SourceCellTrace vals base initial ⟨0,(b,n,PUnit.unit)⟩
      (order.map fun split => (⟨b,split⟩ : Program h))
      (ofFloat h.outputType before) (ofFloat h.outputType after) := by
  induction trace with
  | nil value => exact .nil _
  | cons split rest before middle after atomic tail ih =>
    apply PerAddressProgramIO.SourceCellTrace.cons_of_programs
      (kernelIO) vals base initial ⟨0,(b,n,PUnit.unit)⟩ ⟨b,split⟩ _
      (ofFloat h.outputType before) (ofFloat h.outputType middle) (ofFloat h.outputType after)
      (IOPrograms.programs M P h indices scale orders observe vals domain base binding)
      (IO.program_mem h ⟨b,split⟩) _ ih
    intro final ran
    have result := IOLocality.read_result M P h indices scale orders observe vals domain base binding
      ((kernelIO).toMemoryIO.updateOutput initial ⟨0,(b,n,PUnit.unit)⟩ (ofFloat h.outputType before))
      ⟨b,split⟩ b n final ran
    rw [if_pos ⟨rfl,active⟩,MemoryIO.updateOutput_same,IOPrograms.to_from_float] at result
    rw [atomic]
    exact result

/-- The complete old run, including empty sentinel traces, has the same
output cells as the shared source-derived trace interface. -/
theorem of_run (vals : (kernelIO).Inputs) (domain : IO.domain h indices vals)
    (base : State α) (binding : (kernelIO).InputCells vals base) (initial : (kernelIO).RawOutputs)
    (output : (kernelIO).OutputCells vals initial base)
    (run : FPWorkset.Run M P h orders (IOValues.data h indices scale vals))
    (b : Fin h.batches) (n : Fin h.N) :
    (kernelIO).SourceCellTrace vals base initial ⟨0,(b,n,PUnit.unit)⟩
      ((kernelIO).order vals 0 (b,n,PUnit.unit))
      (initial ⟨0,(b,n,PUnit.unit)⟩) (ofFloat h.outputType (run.output b n)) := by
  rw [initial_value M P h indices scale orders observe vals base binding initial output b n]
  have trace := run.cells b n
  change Trace _ (if indices b.val = -1 then [] else orders.order b n) _ _ at trace
  change (kernelIO).SourceCellTrace vals base initial ⟨0,(b,n,PUnit.unit)⟩
    (if indices b.val = -1 then [] else (orders.order b n).map fun split => ⟨b,split⟩) _ _
  by_cases sentinel : indices b.val = -1
  · rw [if_pos sentinel] at trace ⊢
    have same := trace.fold
    simp only [List.foldl_nil] at same
    rw [same]
    exact .nil _
  · rw [if_neg sentinel] at trace ⊢
    exact of_local_trace M P h indices scale orders observe vals domain base binding initial b n sentinel trace

#axiomsClean initial_value
#axiomsClean of_local_trace
#axiomsClean of_run
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.IOTrace
