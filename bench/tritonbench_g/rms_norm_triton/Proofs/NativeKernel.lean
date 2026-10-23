import bench.tritonbench_g.rms_norm_triton.Proofs.KernelBinding
import bench.tritonbench_g.rms_norm_triton.Proofs.ObservedMath

namespace VeriTile.Bench.TritonBenchG.RmsNormTriton.Observed
open VeriTile.Triton FP.Structural Whole
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

/-- Flat output of the independent native recurrence. Primitive operations,
conversion of literals, runtime epsilon bits and the lane-specific native
reduction tree are model parameters. They are not experimentally assumed
numerical identities or callbacks for an entire RMSNorm row. -/
def nativeFormula (M : NativeModel α) (profile : ReductionFamily) (N B : Nat)
    (x w : TensorView) (entry : Memory α) (k : Nat) : α :=
  reference M profile N B (fun j => x.read entry ((k/N)*N+j))
    (w.read entry) (k%N)

/-- Success and complete physical memory equality for the actual DSL source,
projected through local native instruction lowering. Arbitrary original
strides and storage offsets remain in `x.read` and `w.read`. -/
theorem native_kernel_result (M : NativeModel α) (profile : ReductionFamily) (eps : ℝ)
    (a : Prepared (Term α) input weight (entryGraph entry) preexisting)
    (domain : InputDomain input weight normalized preexisting)
    (h : TensorBindings a x w xc wc y) (schedule : Schedule input normalized)
    (initial : Memory α) (heap : a.output.memory = entryGraph initial) :
    ∃ final, a.entryRun graphAlgebra normalized eps schedule = some final ∧
      KernelResult (nativeFormula M profile (columns normalized) (nextPower (columns normalized)) x w entry)
        y initial (projectMemory M profile (columns normalized) a.output.tensor.base (a.commit final)) ∧
      projectMemory M profile (columns normalized) a.output.tensor.base (a.commit final) =
        referenceMemory (nativeFormula M profile (columns normalized) (nextPower (columns normalized)) x w entry) y initial := by
  obtain ⟨final,ran,result⟩ := prepared_tensor_result a graphAlgebra normalized eps domain h schedule
  have projected : KernelResult
      (nativeFormula M profile (columns normalized) (nextPower (columns normalized)) x w entry)
      y initial (projectMemory M profile (columns normalized) a.output.tensor.base (a.commit final)) := by
    constructor
    · intro k hk
      have offset := a.output_offset h k hk
      have lane : (y.offset k-a.output.tensor.base)%columns normalized = k%columns normalized := by
        rw [offset,Nat.add_sub_cancel_left]
      have value := congrArg (interpret M profile (k%columns normalized)) (result.1 k hk)
      rw [show w.read (entryGraph entry) = (fun j => Term.input (w.read entry j)) from rfl] at value
      simpa only [TensorView.read,projectMemory,lane,entryGraph,Prepared.layout,
        nativeFormula,source_graph_lowering] using value
    · intro allocation offset outside
      have unchanged := result.2 allocation offset outside
      rw [heap] at unchanged
      exact congrArg (interpret M profile ((offset-a.output.tensor.base)%columns normalized)) unchanged
  exact ⟨final,ran,projected,kernelResult_memory h.output a.output.contiguous projected⟩

/-- The independent recurrence also has the requested exact-real meaning. No
such equality is claimed for the rounded native primitives. -/
theorem nativeFormula_real (eps : ℝ) (profile : ReductionFamily) (N B : Nat)
    (complete : N ≤ B) (x w : TensorView) (entry : Memory ℝ) (k : Nat) :
    nativeFormula (realModel eps) profile N B x w entry k =
      realFormula N eps (fun j => x.read entry ((k/N)*N+j)) (w.read entry) (k%N) :=
  reference_real eps profile N B complete _ _ _

end
#guard_msgs (drop info) in
#specNonCircular nativeFormula avoiding [lower,interpret,rowValue,sourceRun,Prepared.entryRun]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsNormTriton.Observed
