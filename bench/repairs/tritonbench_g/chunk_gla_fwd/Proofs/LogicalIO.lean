import bench.repairs.tritonbench_g.chunk_gla_fwd.IO
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.RealImplements

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.RealIO.LogicalIO
open VeriTile Triton
open scoped VeriTile.Triton.LogicalIO
noncomputable section

theorem producer_full_judgment_iff (c : Host.Dimensions) (scale : ℝ)
    (f : ProducerInputs c → Fin c.heads → Fin c.T → Fin c.BT → ℝ) :
    (producer c scale ⊨ f) ↔ (RealIO.producer c scale).Implements
      (fun raw => (outputs3 c.heads c.T c.BT).symm (f (producerInputs c raw))) := Iff.rfl

theorem consumer_full_judgment_iff (c : Host.Dimensions) (scale : ℝ)
    (f : ConsumerInputs c → Fin c.heads → Fin c.T → Fin c.V → ℝ) :
    (consumer c scale ⊨ f) ↔ (RealIO.consumer c scale).Implements
      (fun raw => (outputs3 c.heads c.T c.V).symm (f (consumerInputs c raw))) := Iff.rfl

theorem producer_correct (c : Host.Dimensions) (scale : ℝ) :
    producer c scale ⊨ (fun values head time column =>
      Logical.attention c.T c.K c.BT (producerArrays c values) scale head.val time.val column.val) := by
  apply (producer_full_judgment_iff c scale _).mpr
  have target_eq : (fun raw => (outputs3 c.heads c.T c.BT).symm
      (fun head time column => Logical.attention c.T c.K c.BT
        (producerArrays c (producerInputs c raw)) scale head.val time.val column.val)) = RealIO.attention c scale := by
    funext raw channel j
    change Logical.attention c.T c.K c.BT (producerArrays c (producerInputs c raw)) scale
      j.divNat.divNat.val j.divNat.modNat.val j.modNat.val = _
    unfold producerArrays
    rw [Equiv.symm_apply_apply]
    rfl
  exact (congrArg (RealIO.producer c scale).Implements target_eq).mpr
    (RealImplements.producer_correct c scale)

theorem consumer_correct (c : Host.Dimensions) (scale : ℝ) :
    consumer c scale ⊨ (fun values head time value =>
      KernelReal.consumerFormula c (consumerArrays c values) scale head time value) := by
  apply (consumer_full_judgment_iff c scale _).mpr
  have target_eq : (fun raw => (outputs3 c.heads c.T c.V).symm
      (fun head time value => KernelReal.consumerFormula c
        (consumerArrays c (consumerInputs c raw)) scale head time value)) = RealIO.output c scale := by
    funext raw channel j
    change KernelReal.consumerFormula c (consumerArrays c (consumerInputs c raw)) scale
      j.divNat.divNat j.divNat.modNat j.modNat = _
    unfold consumerArrays
    rw [Equiv.symm_apply_apply]
    rfl
  exact (congrArg (RealIO.consumer c scale).Implements target_eq).mpr
    (RealImplements.consumer_correct c scale)

#axiomsClean producer_full_judgment_iff
#axiomsClean consumer_full_judgment_iff
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.RealIO.LogicalIO
