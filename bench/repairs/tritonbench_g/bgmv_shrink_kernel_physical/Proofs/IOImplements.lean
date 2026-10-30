import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOTrace

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.IOImplements
open VeriTile.Triton FP.Structural FPValues
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset
noncomputable section
set_option maxHeartbeats 1500000
variable {α γ : Type} [Inhabited α]
variable (M : Algebra α) (P : Primitives α) (h : Call)
    (indices : Nat → Int) (scale : α) (orders : Orders h) (observe : α → γ)
local notation "kernelIO" => IO.programIO M P h indices scale orders observe

def channels (run : FPWorkset.Run M P h orders (IOValues.data h indices scale vals)) :
    ChannelValues α (IO.outputs h) :=
  Fin.cases (fun j => ofFloat h.outputType (run.output j.1 j.2.1)) (fun i => Fin.elim0 i)

def results (run : FPWorkset.Run M P h orders (IOValues.data h indices scale vals)) :
    (kernelIO).RawOutputs := fun key => channels M P h indices scale orders run key.1 key.2

theorem map_float (d : FloatDType) (value : α) :
    mapCell observe (Cell.mk d.toTileDType (ofFloat d value)) =
      Cell.mk d.toTileDType (ofFloat d (observe value)) := by
  cases d <;> rfl

/-- Every old runtime linearization has exactly these source-derived output
traces on the caller's bound input memory. No chosen run is an IO premise. -/
theorem realizes_run (vals : (kernelIO).Inputs) (domain : IO.domain h indices vals)
    (base : State α) (binding : (kernelIO).InputCells vals base) (initial : (kernelIO).RawOutputs)
    (output : (kernelIO).OutputCells vals initial base)
    (run : FPWorkset.Run M P h orders (IOValues.data h indices scale vals)) :
    ∀ key, (kernelIO).outputActive vals key →
      (kernelIO).SourceCellTrace vals base initial key ((kernelIO).order vals key.1 key.2)
        (initial key) (results M P h indices scale orders observe run key) := by
  rintro ⟨o,j⟩ _
  have ho : o=0 := Fin.eq_zero o
  subst o
  obtain ⟨b,n,u⟩ := j
  cases u
  exact IOTrace.of_run M P h indices scale orders observe vals domain base binding initial output run b n

theorem implements (f : (kernelIO).Inputs → (kernelIO).Outputs)
    (formula : ∀ vals b n,
      ofFloat h.outputType (observe (FPValues.target M P h orders (IOValues.data h indices scale vals) b n)) =
        f vals 0 (b,n,PUnit.unit)) :
    (kernelIO).ImplementsOnCells (IO.domain h indices) f := by
  intro vals domain base binding initial output
  let run := CanonicalTrace.run M P h orders (IOValues.data h indices scale vals)
  let finalValues := results M P h indices scale orders observe run
  refine ⟨IOGeometry.schedule_complete M P h indices scale orders observe vals,
    IOPrograms.programs M P h indices scale orders observe vals domain base binding,
    IOLocality.locality M P h indices scale orders observe vals domain base binding,
    IOLocality.coverage M P h indices scale orders observe vals domain base binding,
    finalValues,realizes_run M P h indices scale orders observe vals domain base binding initial output run,?_,
    (kernelIO).toMemoryIO.patchOutputs_frame vals finalValues base⟩
  intro o j _
  have ho : o=0 := Fin.eq_zero o
  subst o
  obtain ⟨b,n,u⟩ := j
  cases u
  have cell := (kernelIO).toMemoryIO.patchOutputs_read vals
    (IOGeometry.output_unique M P h indices scale orders observe vals) finalValues base
      ⟨0,(b,n,PUnit.unit)⟩ trivial
  change ((kernelIO).patchOutputs vals finalValues base).mem "Y" (outputOffset h b.val n.val) =
    Cell.mk h.outputType.toTileDType (ofFloat h.outputType
      (FPValues.target M P h orders (IOValues.data h indices scale vals) b n)) at cell
  change mapCell observe (((kernelIO).patchOutputs vals finalValues base).mem "Y" (outputOffset h b.val n.val)) =
    Cell.mk h.outputType.toTileDType (f vals 0 (b,n,PUnit.unit))
  rw [cell,map_float,formula]

#axiomsClean realizes_run
#axiomsClean implements
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.IOImplements
