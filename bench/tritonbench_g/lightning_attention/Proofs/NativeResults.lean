import bench.tritonbench_g.lightning_attention.Proofs.NativeMaterialization

namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.NativeResults
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.LightningAttention.Native (Word Instructions Plan)
open NativeLocality NativeCoverage NativeMaterialization
noncomputable section
set_option maxHeartbeats 6000000

theorem forward (ops : Instructions) (plan : Plan) (d : Host.Dimensions)
    (vals : (NativeIO.forward ops plan d).Inputs) (memory : Memory Word)
    (bound : (NativeIO.forward ops plan d).InputCells vals memory) :
    ∃ final, runMaterializedSchedule ((NativeIO.forward ops plan d).schedule vals) memory = some final ∧
      (NativeIO.forward ops plan d).OutputBinding vals (NativeReference.forward ops plan d vals) final ∧
      (NativeIO.forward ops plan d).Frame vals memory final := by
  obtain ⟨final,ran,values,frame⟩ := grid_result ops plan .forward d (grid .forward d)
    (NativeReference.forwardScope d)
    (fun pid hp => forwardCovered d pid ((mem_grid _ _ _).mp hp))
    memory (NativeReference.forwardWords d vals) (forward_binding ops plan d vals memory bound)
  refine ⟨final,ran,?_,?_⟩
  · intro o j _
    fin_cases o
    have hv := values "output" (NativeIO.paddedIndex d j) (forward_output_scope d j)
    change mapCell id (final "output" (NativeIO.paddedIndex d j)) = _
    rw [hv]
    rfl
  · intro r o outside
    exact frame r o (fun pid hp => forward_outside ops plan d vals r o outside pid
      ((mem_grid _ _ _).mp hp))

theorem backward (ops : Instructions) (plan : Plan) (d : Host.Dimensions)
    (vals : (NativeIO.backward ops plan d).Inputs) (memory : Memory Word)
    (bound : (NativeIO.backward ops plan d).InputCells vals memory) :
    ∃ final, runMaterializedSchedule ((NativeIO.backward ops plan d).schedule vals) memory = some final ∧
      (NativeIO.backward ops plan d).OutputBinding vals (NativeReference.backward ops plan d vals) final ∧
      (NativeIO.backward ops plan d).Frame vals memory final := by
  obtain ⟨middle,intraRan,intraValues,intraFrame⟩ := grid_result ops plan .intra d (grid .intra d)
    (NativeReference.backwardScope d)
    (fun pid hp => backwardCovered d pid.head ((mem_grid _ _ _).mp hp).1)
    memory (NativeReference.backwardWords d vals) (backward_binding ops plan d vals memory bound)
  obtain ⟨final,interRan,values,frame⟩ := grid_result ops plan .inter d (grid .inter d)
    (NativeReference.backwardScope d)
    (fun pid hp => backwardCovered d pid.head ((mem_grid _ _ _).mp hp).1)
    middle (Native.gridReference ops plan .intra d (grid .intra d) (NativeReference.backwardWords d vals))
    intraValues
  refine ⟨final,?_,?_,?_⟩
  · change runMaterializedSchedule (NativeIO.schedule ops plan .intra d ++
      NativeIO.schedule ops plan .inter d) memory = _
    simp only [runMaterializedSchedule_append,NativeIO.schedule,intraRan,Option.bind_some]
    exact interRan
  · intro o j _
    have hv := values _ _ (backward_output_scope d o j)
    fin_cases o
    · change final "queryGradient" (denseIndex j) = _ at hv
      change mapCell id (final "queryGradient" (denseIndex j)) = _
      rw [hv]
      rfl
    · change final "keyGradient" (denseIndex j) = _ at hv
      change mapCell id (final "keyGradient" (denseIndex j)) = _
      rw [hv]
      rfl
    · change final "valueGradient" (denseIndex j) = _ at hv
      change mapCell id (final "valueGradient" (denseIndex j)) = _
      rw [hv]
      rfl
  · intro r o outside
    rw [frame r o (fun pid hp => backward_outside ops plan d vals r o outside .inter
      (Or.inr rfl) pid ((mem_grid _ _ _).mp hp))]
    exact intraFrame r o (fun pid hp => backward_outside ops plan d vals r o outside .intra
      (Or.inl rfl) pid ((mem_grid _ _ _).mp hp))

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.NativeResults

