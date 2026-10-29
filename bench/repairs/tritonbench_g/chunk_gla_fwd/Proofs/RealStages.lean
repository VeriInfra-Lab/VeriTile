import bench.repairs.tritonbench_g.chunk_gla_fwd.IO
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.RealGridBounds

/-! The shared finite-fill/source trace reconstructs the original complete
ordinary-grid producer and consumer executions. The existing mathematical
readback and raw-cell frame proofs are reused without a serial-grid substitute.
-/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.RealStages
open VeriTile Triton RealIO
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000

structure ProducerExtents (c : Host.Dimensions) (bounds : RegionBounds) : Prop where
  query : c.heads*c.T*c.K≤bounds "q"
  key : c.heads*c.T*c.K≤bounds "k"
  gate : c.heads*c.T*c.K≤bounds "g"
  attention : Host.attentionSize c≤bounds "attention"
  scratch : ¬c.K≤256 → Host.scratchSize c≤bounds "scratch"

structure ConsumerExtents (c : Host.Dimensions) (bounds : RegionBounds) : Prop where
  query : c.heads*c.T*c.K≤bounds "q"
  gate : c.heads*c.T*c.K≤bounds "g"
  value : Host.outputSize c≤bounds "v"
  history : c.heads*c.NT*c.K*c.V≤bounds "h"
  attention : Host.attentionSize c≤bounds "attention"
  output : Host.outputSize c≤bounds "o"

theorem attention_fill (c : Host.Dimensions) (s : BlockState) :
    (zeroAttention c).apply s=RealWhole.attentionInit c s := by
  simp [zeroAttention,AlgorithmFill.apply,AlgorithmFill.Contains,
    RealWhole.attentionInit,Allocation.zeroRange]

theorem scratch_fill (c : Host.Dimensions) (s : BlockState) :
    (zeroScratch c).apply (RealWhole.attentionInit c s)=Allocation.initialized c.producer s := by
  simp [zeroScratch,AlgorithmFill.apply,AlgorithmFill.Contains,Host.scratchSize,Host.Dimensions.NK,
    RealWhole.attentionInit,Allocation.zeroRange,Allocation.initialized,Host.attentionSize,Host.Dimensions.producer]
  rfl

theorem grid_undef {k : Kernel} {g : Grid} {s t : BlockState}
    (run : Kernel.GridLaunchedOrdinary k g s t) : t.undef=s.undef := by
  rw [run.h_final]
  rfl

theorem producer_trace (c : Host.Dimensions) (scale : ℝ) (s : BlockState)
    (hu : s.undef=fun _ _=>0) :
    ∃ final,∃ trace : AlgorithmFillExecution (producerStages c scale) s final,
      (∀ bounds,ProducerExtents c bounds → trace.Safe bounds) ∧
      (∀ (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT),
        final.readMem "attention" (h.val*c.T*c.BT+t.val*c.BT+j.val)=
          Logical.attention c.T c.K c.BT (InterGrid.inputs s) scale h.val t.val j.val) ∧
      BlockState.WriteWithin (KernelReal.footprint .producer c) s final := by
  have complete := RealWhole.producer c scale s hu
  by_cases small : c.K≤256
  · rw [if_pos small] at complete
    obtain ⟨final,⟨runs⟩,values,frame⟩ := complete
    rw [producerStages,if_pos small]
    let trace : AlgorithmFillExecution
        [.fill (zeroAttention c),
          .source (.ordinary (interSource c scale) (ProducerGeometry.grid c.producer)),
          .source (.ordinary (intraSource c scale) (DiagonalGeometry.grid c.producer))] s final :=
      .cons (attention_fill c s).symm
        (.cons ⟨runs.interLaunch⟩ (.cons ⟨runs.intraLaunch⟩ (.nil final)))
    refine ⟨final,trace,?_,values,?_⟩
    · intro bounds hb
      have midU : runs.inter.undef=fun _ _=>0 := (grid_undef runs.interLaunch).trans hu
      refine ⟨?_,?_,?_,trivial⟩
      · intro i hi
        simpa only [zeroAttention,Nat.zero_add] using hi.trans_le hb.attention
      · intro idx
        exact RealGridBounds.inter_safe c.producer scale (RealWhole.attentionInit c s) hu idx bounds
          hb.query hb.key hb.gate hb.attention
      · intro idx
        exact RealGridBounds.intra_safe c.producer c.intraBK scale _ midU idx bounds
          hb.query hb.key hb.gate hb.attention
    · simpa only [KernelReal.footprint,if_pos small] using frame
  · rw [if_neg small] at complete
    obtain ⟨final,⟨runs⟩,values,frame⟩ := complete
    rw [producerStages,if_neg small]
    let trace : AlgorithmFillExecution
        [.fill (zeroAttention c),.fill (zeroScratch c),
          .source (.ordinary (interSource c scale) (ProducerGeometry.grid c.producer)),
          .source (.ordinary (splitSource c scale) (SplitGeometry.grid c.split)),
          .source (.ordinary (mergeSource c) (DiagonalGeometry.grid c.split))] s final :=
      .cons (attention_fill c s).symm
        (.cons (scratch_fill c s).symm
          (.cons ⟨runs.interLaunch⟩ (.cons ⟨runs.splitLaunch⟩ (.cons ⟨runs.mergeLaunch⟩ (.nil final)))))
    refine ⟨final,trace,?_,values,?_⟩
    · intro bounds hb
      have midU : runs.inter.undef=fun _ _=>0 := (grid_undef runs.interLaunch).trans hu
      have splitU : runs.split.undef=fun _ _=>0 := (grid_undef runs.splitLaunch).trans midU
      refine ⟨?_,?_,?_,?_,?_,trivial⟩
      · intro i hi
        simpa only [zeroAttention,Nat.zero_add] using hi.trans_le hb.attention
      · intro i hi
        simpa only [zeroScratch,Nat.zero_add] using hi.trans_le (hb.scratch small)
      · intro idx
        exact RealGridBounds.inter_safe c.producer scale (Allocation.initialized c.producer s) hu idx bounds
          hb.query hb.key hb.gate hb.attention
      · intro idx
        exact RealGridBounds.split_safe c.split scale _ midU idx bounds hb.query hb.key hb.gate (hb.scratch small)
      · intro idx
        exact RealGridBounds.merge_safe c.split _ splitU idx bounds (hb.scratch small) hb.attention
    · simpa only [KernelReal.footprint,if_neg small] using frame

theorem consumer_trace (c : Host.Dimensions) (scale : ℝ) (s : BlockState)
    (hu : s.undef=fun _ _=>0) :
    ∃ final,∃ trace : AlgorithmFillExecution
      [.source (.ordinary (outputSource c scale) (OutputGrid.grid c.consumer))] s final,
      (∀ bounds,ConsumerExtents c bounds → trace.Safe bounds) ∧
      (∀ (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V),
        final.readMem "o" (h.val*c.T*c.V+t.val*c.V+v.val)=
          KernelReal.consumerFormula c (LocalReal.outputInputs OutputGrid.regions s) scale h t v) ∧
      BlockState.WriteWithin (KernelReal.footprint .consumer c) s final := by
  obtain ⟨final,runs,values,frame⟩ := RealWhole.consumer c scale s hu
  let trace : AlgorithmFillExecution
      [.source (.ordinary (outputSource c scale) (OutputGrid.grid c.consumer))] s final :=
    .cons runs (.nil final)
  refine ⟨final,trace,?_,?_,frame⟩
  · intro bounds hb
    refine ⟨?_,trivial⟩
    intro idx
    exact RealGridBounds.output_safe c.consumer scale s hu idx bounds
      hb.query hb.gate hb.value hb.history hb.attention hb.output
  · intro h t v
    have value := values h t v
    rw [KernelReal.consumer_formula] at value
    simpa only [OutputGrid.returned,Host.Dimensions.consumer,Nat.mul_assoc] using value

#axiomsClean producer_trace
#axiomsClean consumer_trace
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.RealStages
