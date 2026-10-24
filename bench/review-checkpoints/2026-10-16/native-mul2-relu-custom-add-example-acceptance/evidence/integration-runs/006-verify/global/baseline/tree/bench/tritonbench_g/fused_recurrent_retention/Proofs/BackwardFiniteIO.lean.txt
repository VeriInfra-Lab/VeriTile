import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardIO
import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardFlatExecution

/-! Successful flattened execution and all three independent gradient outputs,
with the full tagged-cell frame. Initial undefined backing storage is irrelevant
to this fully masked source; no final-pointer no-underflow premise is imposed. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardIO
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1200000
set_option linter.unusedSimpArgs false

theorem source_correct (A : FlatAlloc) (ha : A.Disjoint)
    (hc : ∀ region, region ∉ A.regions → A.extent region = 0)
    (p : Layout) (r : BackwardRegions) (hasSeed swapped : Bool) (initial : BlockState)
    (hd : BackwardReal.Disjoint r)
    (hb : BackwardSafety.Bounds p (launch initial) r hasSeed A.extent) :
    ∃ final, exec (A.flattenKernel (BackwardSource.kernel p r hasSeed swapped).toAlgKernel)
      (A.flattenState initial) = some final ∧ Post A p r hasSeed initial final := by
  let initial0 : BlockState := {initial with undef := fun _ _ => 0}
  obtain ⟨regionFinal,he,hp⟩ := BackwardReal.source_correct p r hasSeed swapped initial0 hd
  obtain ⟨flat,hflat,hobs⟩ := BackwardFlatExecution.executes A ha hc p r hasSeed swapped
    initial0 regionFinal hd hb rfl he
  have hpost := flatten_post A ha hc p r hasSeed initial0 regionFinal hb hp
  refine ⟨flat,hflat,?_,?_,?_,?_⟩
  · intro t k ht hk
    rw [hobs.mem]
    exact hpost.dq t k ht hk
  · intro t k ht hk
    rw [hobs.mem]
    exact hpost.dk t k ht hk
  · intro t j ht hj
    rw [hobs.mem]
    exact hpost.dv t j ht hj
  · intro region off hn
    rw [hobs.mem]
    exact hpost.frame region off hn

#axiomsClean source_correct
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardIO
