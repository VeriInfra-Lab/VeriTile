import bench.repairs.tritonbench_g.token_softmax_bloom.Kernels
import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.Grid
import bench.tritonbench_g.triton_softmax.Proofs.IndexRewrite

namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom.KernelRewrite
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 16000

def program (k : ComputeKernel) : Program := ⟨k,[]⟩

/-- Equal integer offsets preserve the signed-metadata source load. -/
@[spec_rule] theorem op_load_region {α} [Inhabited α] (M : FP.Structural.Algebra α)
    (s : FP.Structural.State α) (before : List ComputeStmt)
    (r : Region .nat) (a b : Op .nat [])
    (same : OpEquivalent M s before a b) :
    OpEquivalent M s before (.load .nat (.region r a) .none) (.load .nat (.region r b) .none) := by
  intro t ht
  simp only [FP.Structural.evalOp]
  rw [same t ht]

open TokenSoftmaxLlama.Strided

def candidateKernel (p : Parameters) := indexCommutedSource .fp32 p.input p.start p.length p.output
  p.startStride p.lengthStride p.inputHeadStride p.inputTokenStride p.outputHeadStride p.outputTokenStride p.width

/-- Exactly one integer metadata-address product is commuted in the complete FP32 program. -/
theorem equivalent (p : Parameters) :
    program (kernel .fp32 p) ≡[([] : Spec.Assumptions FP.GuardedFragment)]
      program (candidateKernel p) := by
  equiv_decompose; fp_prove

theorem execution {A : Type} [Inhabited A] (algebra : FP.Structural.Algebra A)
    (p : Parameters) (s : FP.Structural.State A) :
    FP.Structural.exec (engine algebra) (kernel .fp32 p) s=
      FP.Structural.exec (engine algebra) (candidateKernel p) s := by
  change Contextual algebra s ((kernel .fp32 p).surfaceBody.take 3)
    ((kernel .fp32 p).surfaceBody.drop 4)
    (.assign .nat [] "cur_batch_seq_len" (.alg (.load .nat (.region p.length
      (.mul .nat .nil (.ref .nat [] "cur_batch") (.constNat p.lengthStride))) .none)))
    (.assign .nat [] "cur_batch_seq_len" (.alg (.load .nat (.region p.length
      (.mul .nat .nil (.constNat p.lengthStride) (.ref .nat [] "cur_batch"))) .none)))
  fp_prove

def candidateLaunch {α} [Inhabited α] (M : FP.Structural.Algebra α) (p : Parameters)
    (pid : Grid.PID) (m : Grid.Memory α) : Option (Grid.Memory α) :=
  (FP.Structural.exec M (candidateKernel p) (Grid.state m pid)).map FP.Structural.State.mem

def candidateRun {α} [Inhabited α] (M : FP.Structural.Algebra α) (p : Parameters) :
    List Grid.PID → Grid.Memory α → Option (Grid.Memory α)
  | [], m => some m
  | pid::rest, m => (candidateLaunch M p pid m).bind (candidateRun M p rest)

theorem launch_execution {α} [Inhabited α] (M : FP.Structural.Algebra α) (p : Parameters)
    (pid : Grid.PID) (m : Grid.Memory α) :
    Grid.launch (engine M) p pid m=candidateLaunch (engine M) p pid m := by
  exact congrArg (Option.map FP.Structural.State.mem) (execution M p (Grid.state m pid))

theorem grid_execution {α} [Inhabited α] (M : FP.Structural.Algebra α) (p : Parameters)
    (work : List Grid.PID) (m : Grid.Memory α) :
    Grid.run (engine M) p work m=candidateRun (engine M) p work m := by
  induction work generalizing m with
  | nil => rfl
  | cons pid rest ih =>
    simp only [Grid.run,candidateRun,←launch_execution]
    congr 1
    funext next
    exact ih next

#axiomsClean grid_execution
#axiomsClean equivalent
#axiomsClean execution
end
end VeriTile.Bench.Repairs.TokenSoftmaxBloom.KernelRewrite
