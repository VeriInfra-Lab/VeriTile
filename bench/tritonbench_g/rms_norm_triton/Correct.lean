import bench.tritonbench_g.rms_norm_triton.Proofs.KernelIndex
import bench.tritonbench_g.rms_norm_triton.Proofs.KernelBinding

namespace VeriTile.Bench.TritonBenchG.RmsNormTriton
open VeriTile.Triton FP.Structural Whole
noncomputable section

/-- Every logical output of the original typed kernel equals the independent
RMSNorm formula, and its complete physical memory is precisely that output
update. Copy API laws connect all reads to the original Tensor views. -/
theorem rms_norm_kernel_correct
    (a : Prepared ℝ input weight entry preexisting) (eps : ℝ)
    (domain : InputDomain input weight normalized preexisting)
    (binding : TensorBindings a x w xc wc y) (schedule : Schedule input normalized)
    (finite : FiniteAdmission a normalized) (physical : PhysicalBinding finite x w xc wc y) :
    Spec.Real (KernelPointers xc wc y ∧ IndexCorrespondence a normalized finite ∧
      ActualSourceAddresses a normalized finite xc wc y ∧
      ∃ final, a.entryRun realAlgebra normalized eps schedule = some final ∧
        KernelResult (fun k => realFormula (columns normalized) eps
          (fun j => x.read entry ((k/columns normalized)*columns normalized+j))
          (w.read entry) (k%columns normalized)) y a.output.memory (a.commit final) ∧
        a.commit final = referenceMemory (fun k => realFormula (columns normalized) eps
          (fun j => x.read entry ((k/columns normalized)*columns normalized+j))
          (w.read entry) (k%columns normalized)) y a.output.memory) := by
  refine ⟨kernel_pointers a binding,index_correspondence a normalized finite,
    actual_source_addresses a binding domain finite physical,?_⟩
  obtain ⟨final,ran,result⟩ := prepared_tensor_result a realAlgebra normalized eps domain binding schedule
  simp only [rowValue_real,Prepared.layout] at result
  exact ⟨final,ran,result,kernelResult_memory binding.output a.output.contiguous result⟩

#guard_msgs (drop info) in
#specNonCircular realFormula avoiding [kernel,rowValue,sourceRun,Prepared.entryRun]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.RmsNormTriton
