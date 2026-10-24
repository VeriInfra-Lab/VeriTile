import bench.repairs.tritonbench_g.token_attn_mistral.Proofs.Mathematical
import bench.repairs.tritonbench_g.token_attn_mistral.Integers
import bench.repairs.tritonbench_g.token_attn_mistral.HostSchedule
namespace VeriTile.Bench.Repairs.TokenAttnMistral.Host
open VeriTile.Triton FP.Structural Grid
noncomputable section

/-- Complete ordinary FP32 host call under erased real arithmetic: all
metadata/page lookups, the entire unpadded weighted sum, supplied output
identity, None return, address bounds and full memory frame. -/
theorem historical_correct (c : Call) (m : Memory ℝ) (work : List PID)
    (layout : Layout c) (data : Data c m) (storage : Storage c) (integers : IntegerDomain c)
    (legal : (fullGrid (batches c) (heads c)).Perm work) :
    Accesses c m ∧ Formed c m ∧ ∃ result,
      invokeActual Real.algebra Real.sum c work m=some result ∧
      result.returned=none ∧ result.output=c.output ∧
      (∀ (b : Fin (batches c)) (h : Fin (heads c)) (d : Fin c.value.width),
        (result.memory c.output.region ((b.val*heads c+h.val)*c.value.width+d.val)).read .fp32=
          target c m b.val h.val d.val) ∧ Frame c m result.memory := by
  refine ⟨accesses c m layout data storage,formed c m data integers,?_⟩
  rw [invokeActual_eq _ _ _ _ _ layout legal]
  exact whole_real c m layout

#axiomsClean historical_correct
#auditStmt historical_correct
#specNonCircular target avoiding [FP.Structural.exec,Grid.run,invoke,invokeActual]
end
end VeriTile.Bench.Repairs.TokenAttnMistral.Host
