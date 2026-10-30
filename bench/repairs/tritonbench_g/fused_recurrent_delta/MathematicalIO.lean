import bench.repairs.tritonbench_g.fused_recurrent_delta.IO
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelRealAlgebra
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Geometry

/-! Independent Real specifications: the K-dimensional delta recurrence and
its full VJP. The backward premise relates the saved residual input to a
primal sequence only at used times. It imposes no condition after the sequence. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Interface
open VeriTile.Triton FP.Structural
open scoped BigOperators
noncomputable section

def forwardFormula (s : Shape) (f : Flags) (scale : ℝ) (vals : ForwardInputs ℝ s) : ForwardOutputs ℝ s
  | ⟨0,_⟩ => fun j => Math.output scale (primal s f vals j.2.1) j.1.val
  | ⟨1,_⟩ => fun j => Math.residual (Math.state (primal s f vals j.2.1) j.1.val)
      ((primal s f vals j.2.1).k j.1.val) ((primal s f vals j.2.1).x j.1.val)
  | ⟨2,_⟩ => fun j => if hi : j.1.val<s.K then Math.state (primal s f vals j.2.1) s.T ⟨j.1.val,hi⟩ else (0 : ℝ)
  | ⟨n+3,h⟩ => False.elim (by simp [forwardOutputs] at h)

def backwardFormula (s : Shape) (scale : ℝ) (a : Fin s.BBV → Math.Input s.BK)
    (g : Nat → Fin s.BBV → ℝ) (terminal : Fin s.BBV → Math.Vec s.BK)
    (_ : BackwardInputs ℝ s) : BackwardOutputs ℝ s
  | ⟨0,_⟩ => fun j => (∑ v : Fin s.BBV,(Math.gradient scale (a v) (fun t => g t v) (terminal v) s.T j.1.val).q j.2.1 : ℝ)
  | ⟨1,_⟩ => fun j => (∑ v : Fin s.BBV,(Math.gradient scale (a v) (fun t => g t v) (terminal v) s.T j.1.val).k j.2.1 : ℝ)
  | ⟨2,_⟩ => fun j => (Math.gradient scale (a j.2.1) (fun t => g t j.2.1) (terminal j.2.1) s.T j.1.val).x
  | ⟨3,_⟩ => fun j => (Math.gradient scale (a j.2.1) (fun t => g t j.2.1) (terminal j.2.1) s.T j.1.val).beta
  | ⟨4,_⟩ => fun j => (∑ v : Fin s.BBV,(Math.gradient scale (a v) (fun t => g t v) (terminal v) s.T j.1.val).beta : ℝ)
  | ⟨5,_⟩ => fun j => Math.carry scale (a j.2.1) (fun t => g t j.2.1) (terminal j.2.1) s.T 0 j.1
  | ⟨n+6,h⟩ => False.elim (by simp [backwardOutputs] at h)

#specNonCircular forwardFormula avoiding [FP.Structural.exec,FP.Structural.run,KernelProgram.source,KernelReference.forward]
#specNonCircular backwardFormula avoiding [FP.Structural.exec,FP.Structural.run,KernelProgram.source,KernelBackwardReference.reverse]
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Interface
