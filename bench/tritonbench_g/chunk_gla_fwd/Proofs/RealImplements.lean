import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealStages
import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealInputs

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.RealImplements
open VeriTile Triton RealIO
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000

theorem producer_extents (c : Host.Dimensions) (scale : ℝ) (values : (producer c scale).Ctx)
    (s : BlockState) (bounds : RegionBounds) (hb : (producer c scale).InBounds bounds s values) :
    RealStages.ProducerExtents c bounds := by
  constructor
  · exact RealInputs.extent (fun i=>hb.1 (0 : Fin 3) i trivial)
  · exact RealInputs.extent (fun i=>hb.1 (1 : Fin 3) i trivial)
  · exact RealInputs.extent (fun i=>hb.1 (2 : Fin 3) i trivial)
  · exact RealInputs.extent (fun i=>hb.2.1 (0 : Fin 1) i trivial)
  · intro large
    apply RealInputs.extent
    intro i
    have hi : i.val<(producer c scale).sarity (0 : Fin 1) := by
      simpa only [producer,if_neg large] using i.isLt
    exact hb.2.2 (0 : Fin 1) ⟨i.val,hi⟩ trivial

theorem consumer_extents (c : Host.Dimensions) (scale : ℝ) (values : (consumer c scale).Ctx)
    (s : BlockState) (bounds : RegionBounds) (hb : (consumer c scale).InBounds bounds s values) :
    RealStages.ConsumerExtents c bounds := by
  constructor
  · exact RealInputs.extent (fun i=>hb.1 (0 : Fin 5) i trivial)
  · exact RealInputs.extent (fun i=>hb.1 (1 : Fin 5) i trivial)
  · exact RealInputs.extent (fun i=>hb.1 (2 : Fin 5) i trivial)
  · exact RealInputs.extent (fun i=>hb.1 (3 : Fin 5) i trivial)
  · exact RealInputs.extent (fun i=>hb.1 (4 : Fin 5) i trivial)
  · exact RealInputs.extent (fun i=>hb.2.1 (0 : Fin 1) i trivial)

theorem producer_frame (c : Host.Dimensions) (scale : ℝ) (values : (producer c scale).Ctx)
    (s final : BlockState)
    (frame : BlockState.WriteWithin (KernelReal.footprint .producer c)
      (AlgorithmInitialUndef.zero.initial s) final) : (producer c scale).Frame s final values := by
  intro r address output scratch
  apply (frame r address ?_).symm
  unfold KernelReal.footprint
  split
  · rintro ⟨same,inside⟩
    dsimp only at same inside
    subst r
    rcases output (0 : Fin 1) ⟨address,inside⟩ trivial with h|h
    · exact h rfl
    · exact h rfl
  · rename_i large
    rintro (⟨same,inside⟩|⟨same,inside⟩)
    · dsimp only at same inside
      subst r
      rcases output (0 : Fin 1) ⟨address,inside⟩ trivial with h|h
      · exact h rfl
      · exact h rfl
    · dsimp only at same inside
      subst r
      have hi : address<(producer c scale).sarity (0 : Fin 1) := by
        simpa only [producer,if_neg large,Host.scratchSize,Host.Dimensions.NK,
          Producer.splitShape,Host.Dimensions.producer] using inside
      rcases scratch (0 : Fin 1) ⟨address,hi⟩ trivial with h|h
      · exact h rfl
      · exact h rfl

theorem consumer_frame (c : Host.Dimensions) (scale : ℝ) (values : (consumer c scale).Ctx)
    (s final : BlockState)
    (frame : BlockState.WriteWithin (KernelReal.footprint .consumer c)
      (AlgorithmInitialUndef.zero.initial s) final) : (consumer c scale).Frame s final values := by
  intro r address output _
  apply (frame r address ?_).symm
  rintro ⟨same,inside⟩
  dsimp only at same inside
  subst r
  rcases output (0 : Fin 1) ⟨address,inside⟩ trivial with h|h
  · exact h rfl
  · exact h rfl

theorem producer_correct (c : Host.Dimensions) (scale : ℝ) :
    (producer c scale).Implements (attention c scale) := by
  intro values _ initial bound
  obtain ⟨final,trace,safe,valuesCorrect,frame⟩ :=
    RealStages.producer_trace c scale (AlgorithmInitialUndef.zero.initial initial) rfl
  refine ⟨final,trace,?_,?_,producer_frame c scale values initial final frame⟩
  · intro bounds hb
    exact safe bounds (producer_extents c scale values initial bounds hb)
  · intro o i _
    change final.readMem "attention" i.val=attention c scale values o i
    have result := valuesCorrect i.divNat.divNat i.divNat.modNat i.modNat
    rw [RealInputs.flat_coordinate] at result
    exact result.trans (RealInputs.attention_formula c scale values initial bound
      i.divNat.divNat i.divNat.modNat i.modNat)

theorem consumer_correct (c : Host.Dimensions) (scale : ℝ) :
    (consumer c scale).Implements (output c scale) := by
  intro values _ initial bound
  obtain ⟨final,trace,safe,valuesCorrect,frame⟩ :=
    RealStages.consumer_trace c scale (AlgorithmInitialUndef.zero.initial initial) rfl
  refine ⟨final,trace,?_,?_,consumer_frame c scale values initial final frame⟩
  · intro bounds hb
    exact safe bounds (consumer_extents c scale values initial bounds hb)
  · intro o i _
    change final.readMem "o" i.val=output c scale values o i
    have result := valuesCorrect i.divNat.divNat i.divNat.modNat i.modNat
    rw [RealInputs.flat_coordinate] at result
    exact result.trans (RealInputs.consumer_formula c scale values initial bound
      i.divNat.divNat i.divNat.modNat i.modNat)

#axiomsClean producer_correct
#axiomsClean consumer_correct
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.RealImplements
