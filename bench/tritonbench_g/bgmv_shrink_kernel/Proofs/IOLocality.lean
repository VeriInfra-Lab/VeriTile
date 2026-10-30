import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.IOPrograms

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.IOLocality
open VeriTile.Triton FP.Structural FPValues MemoryTransport
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset
noncomputable section
set_option maxHeartbeats 1000000
variable {α γ : Type} [Inhabited α]
variable (M : Algebra α) (P : Primitives α) (h : Call)
    (indices : Nat → Int) (scale : α) (orders : Orders h) (observe : α → γ)
local notation "kernelIO" => IO.programIO M P h indices scale orders observe

/-- A CTA changes only its own active batch row. Its selected result depends
on that address's current value and the read operands, never on other outputs. -/
theorem read_result (vals : (kernelIO).Inputs) (domain : IO.domain h indices vals)
    (base : State α) (binding : (kernelIO).InputCells vals base) (current : (kernelIO).RawOutputs)
    (p : Program h) (b : Fin h.batches) (n : Fin h.N) (final : State α)
    (ran : (IO.invocation M P h scale p).run ((kernelIO).patchOutputs vals current base) = some final) :
    final.mem "Y" (outputOffset h b.val n.val) =
      if p.batch = b ∧ indices b.val ≠ -1 then
        Cell.mk h.outputType.toTileDType (ofFloat _
          (FPWorkset.step M P h (IOValues.data h indices scale vals) b n p.split
            (toFloat h.outputType (current ⟨0,(b,n,PUnit.unit)⟩))))
      else Cell.mk h.outputType.toTileDType (current ⟨0,(b,n,PUnit.unit)⟩) := by
  obtain ⟨actual,actualRun,skip,values,frame⟩ :=
    IOPrograms.program_result M P h indices scale orders observe vals domain base binding current p
  have same : actual=final := Option.some.inj (actualRun.symm.trans ran)
  subst actual
  have patched := (kernelIO).toMemoryIO.patchOutputs_read vals
    (IOGeometry.output_unique M P h indices scale orders observe vals) current base ⟨0,(b,n,PUnit.unit)⟩ trivial
  change ((kernelIO).patchOutputs vals current base).mem "Y" (outputOffset h b.val n.val) =
    Cell.mk h.outputType.toTileDType (current ⟨0,(b,n,PUnit.unit)⟩) at patched
  by_cases row : p.batch = b
  · rcases p with ⟨pb,split⟩
    change pb=b at row
    subst pb
    by_cases active : indices b.val ≠ -1
    · rw [if_pos ⟨rfl,active⟩]
      exact values active n
    · have sentinel : indices b.val = -1 := Classical.not_not.mp active
      rw [skip sentinel]
      simpa only [active,and_false,if_false] using patched
  · rw [if_neg (fun hit => row hit.1)]
    rw [frame]
    · exact patched
    · rintro ⟨_,_,m,address⟩
      have rows := (output_injective h b.val p.batch.val n m address).1
      exact row (Fin.ext rows).symm

theorem locality (vals : (kernelIO).Inputs) (domain : IO.domain h indices vals)
    (base : State α) (binding : (kernelIO).InputCells vals base) :
    (kernelIO).SourceLocality vals base := by
  intro p _ key _ left right same l r leftRun rightRun
  rcases key with ⟨o,j⟩
  have ho : o=0 := Fin.eq_zero o
  subst o
  obtain ⟨b,n,unit⟩ := j
  cases unit
  change left ⟨0,(b,n,PUnit.unit)⟩ = right ⟨0,(b,n,PUnit.unit)⟩ at same
  change l.mem "Y" (outputOffset h b.val n.val) = r.mem "Y" (outputOffset h b.val n.val)
  rw [read_result M P h indices scale orders observe vals domain base binding left p b n l leftRun,
    read_result M P h indices scale orders observe vals domain base binding right p b n r rightRun,same]

theorem coverage (vals : (kernelIO).Inputs) (domain : IO.domain h indices vals)
    (base : State α) (binding : (kernelIO).InputCells vals base) :
    (kernelIO).SourceCoverage vals base := by
  intro key _ p _ absent current final ran
  rcases key with ⟨o,j⟩
  have ho : o=0 := Fin.eq_zero o
  subst o
  obtain ⟨b,n,unit⟩ := j
  cases unit
  change p ∉ (if indices b.val = -1 then [] else
    (List.finRange 64).map (fun split => (⟨b,split⟩ : Program h))) at absent
  change final.mem "Y" (outputOffset h b.val n.val) =
    Cell.mk h.outputType.toTileDType (current ⟨0,(b,n,PUnit.unit)⟩)
  rw [read_result M P h indices scale orders observe vals domain base binding current p b n final ran]
  apply if_neg
  rintro ⟨row,active⟩
  apply absent
  rw [if_neg active]
  exact (IOGeometry.row_mem h b p).mpr row

#axiomsClean locality
#axiomsClean coverage
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.IOLocality
