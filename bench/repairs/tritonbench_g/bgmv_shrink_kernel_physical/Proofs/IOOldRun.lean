import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOProjection

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.IOOldRun
open VeriTile.Triton FP.Structural FPValues
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset
noncomputable section
set_option maxHeartbeats 1500000
variable {α γ : Type} [Inhabited α]
variable (M : Algebra α) (P : Primitives α) (h : Call) (orders : Orders h) (observe : α → γ) (data : Data α)
local notation "kernelIO" => IO.programIO M P h data.indices data.scaling orders observe
local notation "vals" => IOProjection.values h data

def initialChannels : ChannelValues α (IO.outputs h) :=
  Fin.cases (fun j => vals 3 j) (fun i => Fin.elim0 i)

def initial : (kernelIO).RawOutputs := fun key => initialChannels h data key.1 key.2

theorem input_cells : (kernelIO).InputCells vals (entry h 0 0 data) := by
  intro i j _
  fin_cases i <;>
    simp only [IO.programIO,IO.inputs,IOProjection.values,
      entry,State.setReg,Data.cells,IOBinding.map_id] <;> rfl

theorem output_cells : (kernelIO).OutputCells vals (initial M P h orders observe data) (entry h 0 0 data) := by
  rintro ⟨o,j⟩ _
  have ho : o=0 := Fin.eq_zero o
  subst o
  rfl

/-- The old complete runtime trace has the same typed endpoints in the shared
source execution relation on its original input memory and undef values. -/
theorem source_run (valid : ∀ b,b<h.batches → data.indices b=-1 ∨ 0≤data.indices b)
    (original : FPWorkset.Run M P h orders data) (b : Fin h.batches) (n : Fin h.N) :
    (kernelIO).SourceCellTrace vals (entry h 0 0 data) (initial M P h orders observe data)
      ⟨0,(b,n,PUnit.unit)⟩ ((kernelIO).order vals 0 (b,n,PUnit.unit))
      (ofFloat h.outputType (data.output (outputOffset h b.val n.val)))
      (ofFloat h.outputType (original.output b n)) := by
  have domain : IO.domain h data.indices vals := ⟨fun _ => rfl,fun b => valid b.val b.isLt⟩
  exact IOTrace.of_run M P h data.indices data.scaling orders observe vals domain (entry h 0 0 data)
    (input_cells M P h orders observe data) (initial M P h orders observe data)
    (output_cells M P h orders observe data) (IOProjection.run M P h orders data original) b n

/-- The new output patch keeps the old stronger frame, including every
sentinel Y cell; all non-Y and exterior Y raw cells are retained exactly. -/
theorem frame (original : FPWorkset.Run M P h orders data) (r : RegionName) (off : Nat)
    (outside : ¬(r="Y" ∧ ∃ b : Fin h.batches, ∃ n : Fin h.N,
      data.indices b.val ≠ -1 ∧ off=outputOffset h b.val n.val)) :
    ((kernelIO).patchOutputs vals
      (IOImplements.results M P h data.indices data.scaling orders observe (IOProjection.run M P h orders data original))
      (entry h 0 0 data)).mem r off = (entry h 0 0 data).mem r off := by
  classical
  let result := IOImplements.results M P h data.indices data.scaling orders observe (IOProjection.run M P h orders data original)
  by_cases hit : ∃ b : Fin h.batches, ∃ n : Fin h.N, r="Y" ∧ off=outputOffset h b.val n.val
  · obtain ⟨b,n,rfl,rfl⟩ := hit
    have sentinel : data.indices b.val = -1 := by
      by_contra active
      exact outside ⟨rfl,b,n,active,rfl⟩
    have cell := (kernelIO).toMemoryIO.patchOutputs_read vals
      (IOGeometry.output_unique M P h data.indices data.scaling orders observe vals)
      result (entry h 0 0 data) ⟨0,(b,n,PUnit.unit)⟩ trivial
    change ((kernelIO).patchOutputs vals result (entry h 0 0 data)).mem "Y" (outputOffset h b.val n.val) =
      Cell.mk h.outputType.toTileDType (ofFloat h.outputType (original.output b n)) at cell
    rw [cell,FPWorkset.readback]
    simp only [FPValues.target,sentinel,if_true]
    rfl
  · apply (kernelIO).toMemoryIO.patchOutputs_frame
    constructor
    · intro o j _
      have ho : o=0 := Fin.eq_zero o
      subst o
      change r≠"Y" ∨ off≠outputOffset h j.1.val j.2.1.val
      exact or_iff_not_imp_left.mpr (fun region address => hit ⟨j.1,j.2.1,Classical.not_not.mp region,address⟩)
    · intro k
      exact Fin.elim0 k

#axiomsClean source_run
#axiomsClean frame
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.IOOldRun
