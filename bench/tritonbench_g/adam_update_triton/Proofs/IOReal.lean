import bench.tritonbench_g.adam_update_triton.Proofs.Interface
import bench.tritonbench_g.adam_update_triton.Proofs.Real

namespace VeriTile.Bench.Repairs.AdamUpdateTriton.Real
open VeriTile.Triton FP.Structural
open Interface
noncomputable section

theorem io_correct (params : Scalars ℝ) (total n B pid : Nat) :
    (programIO semantics params id id total n B pid).Implements
      (fun values => result
        (fun j => Math.target params .parameter (parameter values j)
          (gradient values j) (momentum values j))
        (fun j => Math.target params .momentum (parameter values j)
          (gradient values j) (momentum values j))) := by
  refine Interface.implements semantics params id id total n B pid
    (fun values output j => Math.target params output (parameter values j)
      (gradient values j) (momentum values j)) ?_
  intro values s hp hg hm
  obtain ⟨final, ran, outputs, frame⟩ := kernel_inputs params total n B pid s
    (parameter values) (gradient values) (momentum values) hp hg hm
  refine ⟨final, ran, ?_, frame⟩
  intro output j active
  rw [outputs output j active]
  rfl

#axiomsClean io_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.Real
