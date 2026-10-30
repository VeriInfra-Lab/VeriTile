import bench.repairs.tritonbench_g.token_softmax_llama.Kernels
import bench.repairs.tritonbench_g.token_softmax_bloom.Proofs.KernelRewrite
import bench.repairs.tritonbench_g.token_softmax_bloom.Proofs.KernelTensor

namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.KernelSource
open VeriTile Triton FP.ProfiledRewrite Strided
open scoped VeriTile.Spec
noncomputable section

/-- The actual Llama source is the quoted program already shared by Bloom. -/
theorem source_is_shared : Strided.source=TokenSoftmaxBloom.source := rfl

/-- No new guard, contiguity restriction, or metadata pin is introduced. -/
theorem domain_is_shared : Checked.InputContract=TokenSoftmaxBloom.Checked.InputContract := rfl

/- Candidate source and invocation retain their names in Kernels.lean. -/

theorem candidate_is_shared : indexCommutedSource=TokenSoftmaxBloom.indexCommutedSource := rfl


abbrev program := TokenSoftmaxBloom.KernelRewrite.program

/-- The complete source/candidate equivalence has an empty numerical context. -/
theorem equivalent (p : Parameters) :
    program (kernel .fp32 p) ≡[([] : Spec.Assumptions FP.GuardedFragment)]
      program (candidateKernel p) :=
  TokenSoftmaxBloom.KernelRewrite.equivalent p

theorem execution {A : Type} [Inhabited A] (algebra : FP.Structural.Algebra A)
    (p : Parameters) (s : FP.Structural.State A) :
    FP.Structural.exec (engine algebra) (kernel .fp32 p) s=
      FP.Structural.exec (engine algebra) (candidateKernel p) s :=
  TokenSoftmaxBloom.KernelRewrite.execution algebra p s

/-- These are executions of the definitionally identical complete candidate. -/
abbrev candidateLaunch := @TokenSoftmaxBloom.KernelRewrite.candidateLaunch
abbrev candidateRun := @TokenSoftmaxBloom.KernelRewrite.candidateRun

theorem candidate_launch_is_quoted {A : Type} [Inhabited A] (algebra : FP.Structural.Algebra A)
    (p : Parameters) (pid : Grid.PID) (m : Grid.Memory A) :
    candidateLaunch algebra p pid m=
      (FP.Structural.exec algebra (candidateKernel p) (Grid.state m pid)).map FP.Structural.State.mem := rfl

theorem grid_execution {A : Type} [Inhabited A] (algebra : FP.Structural.Algebra A)
    (p : Parameters) (work : List Grid.PID) (m : Grid.Memory A) :
    Grid.run (engine algebra) p work m=candidateRun (engine algebra) p work m :=
  TokenSoftmaxBloom.KernelRewrite.grid_execution algebra p work m

#axiomsClean source_is_shared
#axiomsClean domain_is_shared
#axiomsClean candidate_is_shared
#axiomsClean equivalent
#axiomsClean execution
#axiomsClean candidate_launch_is_quoted
#axiomsClean grid_execution
end
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.KernelSource
