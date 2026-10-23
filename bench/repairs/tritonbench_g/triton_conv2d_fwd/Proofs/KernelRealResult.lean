import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelExecution
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelReal

namespace VeriTile.Bench.Repairs.Conv2d.KernelRealResult
open VeriTile.Triton FP.Structural KernelNumeric
noncomputable section

/-- Complete shared-source grid over raw kernel state: independent grouped
cross-correlation, unchanged input regions and every exterior memory cell. -/
theorem correct_real (s : Shape) (valid : Valid s) (flags : Flags)
    (initial : State (KernelNumeric.Value ℝ)) :
    ∃ final, KernelGrid.launch (algebra KernelReal.operations) (halfDot KernelReal.operations)
      (defaultDot KernelReal.operations) s flags initial=some final ∧
      KernelGrid.SameInputs final initial ∧
      (∀ x,s.Active x → final.mem "output" (outputPointer s x)=
        .mk .fp32 (.scalar (mathematical s (memory KernelReal.operations initial) x))) ∧
      (∀ (r : RegionName) a,r≠"output" ∨ s.size≤a → final.mem r a=initial.mem r a) := by
  obtain ⟨final,run,inputs,values,frame⟩ := launch_reference KernelReal.operations s valid flags initial
  refine ⟨final,run,inputs,?_,frame⟩
  intro x active
  rw [values x active,KernelReal.reference_real s _ flags x active]

#axiomsClean correct_real
#specNonCircular mathematical avoiding [KernelSource.kernel,KernelGrid.launch]
end
end VeriTile.Bench.Repairs.Conv2d.KernelRealResult
