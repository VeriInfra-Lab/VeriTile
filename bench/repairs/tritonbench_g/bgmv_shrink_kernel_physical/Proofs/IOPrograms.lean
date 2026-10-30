import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOBinding
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.RebasedSourceSafety
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.TransportComplete

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.IOPrograms
open VeriTile.Triton FP.Structural FPValues MemoryTransport
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 30000
variable {α γ : Type} [Inhabited α]
variable (M : Algebra α) (P : Primitives α) (h : Call)
    (indices : Nat → Int) (scale : α) (orders : Orders h) (observe : α → γ)
local notation "kernelIO" => IO.programIO M P h indices scale orders observe

/-- The IO invocation installs the identical runtime argument, local registers,
program IDs and grid dimensions, while keeping the actual backing memory. -/
theorem initial (vals : (kernelIO).Inputs) (current : (kernelIO).RawOutputs) (base : State α) (p : Program h) :
    (IO.invocation M P h scale p).initial ((kernelIO).patchOutputs vals current base) =
      rebase (entry h p.batch.val p.split.val (IOBinding.currentData M P h indices scale orders observe vals current base))
        ((kernelIO).patchOutputs vals current base).mem := by
  rfl

theorem source_safe (vals : (kernelIO).Inputs) (domain : IO.domain h indices vals)
    (base : State α) (binding : (kernelIO).InputCells vals base) (current : (kernelIO).RawOutputs)
    (p : Program h) (bounds : RegionName → Nat) (bounded : (kernelIO).InBounds bounds vals) :
    (IO.invocation M P h scale p).Safe bounds ((kernelIO).patchOutputs vals current base) := by
  unfold Invocation.Safe
  rw [initial]
  apply RebasedSafety.source_safe M P h p.batch.val p.split.val
    (IOBinding.currentData M P h indices scale orders observe vals current base)
    ((kernelIO).patchOutputs vals current base).mem
    (domain.2 p.batch) (IOBinding.patch_index M P h indices scale orders observe vals domain base binding current p.batch)
  · intro active k hk
    exact IOBinding.patch_input M P h indices scale orders observe vals base binding current p.batch active ⟨k,hk⟩
  · intro active n k hn hk
    exact IOBinding.patch_weight M P h indices scale orders observe vals base binding current p.batch active ⟨n,hn⟩ ⟨k,hk⟩
  · exact bounded.2.1 0 (p.batch,PUnit.unit) trivial
  · intro active k hk
    exact bounded.2.1 1 (p.batch,⟨k,hk⟩,PUnit.unit) active
  · intro active n k hn hk
    exact bounded.2.1 2 (⟨(indices p.batch.val).toNat,IO.selected_inside h indices p.batch⟩,⟨n,hn⟩,⟨k,hk⟩,PUnit.unit)
      ⟨p.batch,active,rfl⟩
  · intro _ n hn
    exact bounded.2.2.1 0 (p.batch,⟨n,hn⟩,PUnit.unit) trivial

theorem to_from_float (d : FloatDType) (x : α) : toFloat d (ofFloat d x) = x := by
  cases d <;> rfl

/-- One complete actual CTA at arbitrary current output cells. The observable
update uses the established source payload; every exterior raw cell is kept. -/
theorem program_result (vals : (kernelIO).Inputs) (domain : IO.domain h indices vals)
    (base : State α) (binding : (kernelIO).InputCells vals base) (current : (kernelIO).RawOutputs) (p : Program h) :
    ∃ final, (IO.invocation M P h scale p).run ((kernelIO).patchOutputs vals current base) = some final ∧
      (indices p.batch.val = -1 → final.mem = ((kernelIO).patchOutputs vals current base).mem) ∧
      (indices p.batch.val ≠ -1 → ∀ n : Fin h.N,
        final.mem "Y" (outputOffset h p.batch.val n.val) = .mk h.outputType.toTileDType
          (ofFloat _ (FPWorkset.step M P h (IOValues.data h indices scale vals) p.batch n p.split
            (toFloat h.outputType (current ⟨0,(p.batch,n,PUnit.unit)⟩))))) ∧
      (∀ (r : RegionName) off, ¬(indices p.batch.val ≠ -1 ∧ r="Y" ∧ ∃ n : Fin h.N, off=outputOffset h p.batch.val n.val) →
        final.mem r off = ((kernelIO).patchOutputs vals current base).mem r off) := by
  let data := IOBinding.currentData M P h indices scale orders observe vals current base
  obtain ⟨final,ran,sentinel,values,frame⟩ := every_program_transport M P h p data
    ((kernelIO).patchOutputs vals current base).mem (domain.2 p.batch)
    (IOBinding.patch_index M P h indices scale orders observe vals domain base binding current p.batch)
    (fun active k hk => IOBinding.patch_input M P h indices scale orders observe vals base binding current p.batch active ⟨k,hk⟩)
    (fun active n k hn hk => IOBinding.patch_weight M P h indices scale orders observe vals base binding current p.batch active ⟨n,hn⟩ ⟨k,hk⟩)
  refine ⟨final,?_,sentinel,?_,frame⟩
  · simpa only [Invocation.run,initial] using ran
  · intro active n
    have result := values active n
    rw [IOBinding.patch_output M P h indices scale orders observe vals current base p.batch n] at result
    have payloadSame := FPReadback.payload_independent M P h data (IOValues.data h indices scale vals)
      rfl rfl rfl p.batch.val (indices p.batch.val).toNat p.split.val
    have indexSame : data.indices = indices := rfl
    rw [indexSame] at result
    rw [payloadSame] at result
    change final.mem "Y" (outputOffset h p.batch.val n.val) =
      Cell.mk h.outputType.toTileDType (ofFloat _ (P.atomic h.outputType
        (toFloat h.outputType (current ⟨0,(p.batch,n,PUnit.unit)⟩))
        (payload M P h (IOValues.data h indices scale vals) p.batch.val (indices p.batch.val).toNat
          p.split.val ⟨n.val,n.isLt.trans_le h.rowsCovered⟩)))
    simpa only [Cell.read,eq_self_iff_true,dite_true,ite_true,IOBinding.float_roundtrip,to_from_float,
      data,IOBinding.currentData,IOValues.data,IOBinding.current_value] using result

/-- Complete-program success, source safety and frame are proved for every
catalog entry, even when the result channel has zero lanes. -/
theorem programs (vals : (kernelIO).Inputs) (domain : IO.domain h indices vals)
    (base : State α) (binding : (kernelIO).InputCells vals base) :
    (kernelIO).Programs vals base := by
  intro current p _
  refine ⟨source_safe M P h indices scale orders observe vals domain base binding current p,?_⟩
  obtain ⟨final,ran,_,_,frame⟩ := program_result M P h indices scale orders observe vals domain base binding current p
  refine ⟨final,ran,?_⟩
  intro r off outside
  apply frame
  rintro ⟨_,region,n,address⟩
  have miss := outside.1 0 (p.batch,n,PUnit.unit) trivial
  exact miss.elim (fun different => different region) (fun different => different address)

#axiomsClean source_safe
#axiomsClean program_result
#axiomsClean programs
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.IOPrograms
