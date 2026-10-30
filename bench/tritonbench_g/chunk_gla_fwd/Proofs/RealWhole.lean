import bench.tritonbench_g.chunk_gla_fwd.Proofs.Allocation
import bench.tritonbench_g.chunk_gla_fwd.Proofs.Admission

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.RealWhole
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

/-- Ordinary zero-initialization of the returned rectangular attention allocation. -/
def attentionInit (c : Host.Dimensions) (s : BlockState) : BlockState :=
  Allocation.zeroRange s "attention" (Host.attentionSize c)

theorem attentionInit_zero (c : Host.Dimensions) (s : BlockState) :
    Producer.ZeroAttention c.producer (attentionInit c s) := by
  intro h t j
  apply Allocation.zeroRange_read
  simpa only [Host.attentionSize,Host.Dimensions.producer,Geometry.attentionAddress,Nat.add_mul,Nat.mul_assoc]
    using Geometry.attention_bound h.isLt t.isLt j.isLt

theorem attentionInit_inputs (c : Host.Dimensions) (s : BlockState) :
    InterGrid.inputs (attentionInit c s)=InterGrid.inputs s := by
  apply StageFrames.inputs_of_frame
  intro r ha hs off
  simp [attentionInit,Allocation.zeroRange,ha]

/-- Successful producer source execution is constructed for the branch selected
by K. The caller supplies only the allocation state, never a kernel result. -/
theorem producer (c : Host.Dimensions) (scale : ℝ) (s : BlockState)
    (hu : s.undef=fun _ _ => 0) :
    if c.K≤256 then
      ∃ result,
        Nonempty (Producer.DirectRun c.producer c.intraBK scale (attentionInit c s) result) ∧
        (∀ (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT),
          result.readMem "attention" (h.val*c.T*c.BT+t.val*c.BT+j.val)=
            Logical.attention c.T c.K c.BT (InterGrid.inputs s) scale h.val t.val j.val) ∧
        BlockState.WriteWithin (Allocation.attention c.producer) s result
    else
      ∃ result,
        Nonempty (Producer.LargeRun c.producer scale (Allocation.initialized c.producer s) result) ∧
        (∀ (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT),
          result.readMem "attention" (h.val*c.T*c.BT+t.val*c.BT+j.val)=
            Logical.attention c.T c.K c.BT (InterGrid.inputs s) scale h.val t.val j.val) ∧
        BlockState.WriteWithin (fun cell => Allocation.attention c.producer cell ∨
          Allocation.scratch (Producer.splitShape c.producer) cell) s result := by
  split
  · let a:=attentionInit c s
    have hau : a.undef=fun _ _ => 0 := hu
    refine ⟨Producer.direct c.producer c.intraBK (Host.intra_covers_K c) scale a hau,
      ⟨Producer.direct_runs c.producer c.intraBK (Host.intra_covers_K c) scale a hau⟩,?_,?_⟩
    · intro h t j
      have value:=Producer.direct_readback c.producer c.intraBK (Host.intra_covers_K c) scale a hau (attentionInit_zero c s) h t j
      have same : InterGrid.inputs a=InterGrid.inputs s := attentionInit_inputs c s
      rw [same] at value
      exact value
    · intro r off hn
      exact (Allocation.zeroRange_frame s "attention" (Host.attentionSize c) r off hn).trans
        (Allocation.direct_frame c.producer c.intraBK (Host.intra_covers_K c) scale a hau r off hn)
  · let a:=Allocation.initialized c.producer s
    have hau : a.undef=fun _ _ => 0 := hu
    refine ⟨Producer.large c.producer scale a hau,
      ⟨Producer.large_runs c.producer scale a hau⟩,?_,?_⟩
    · intro h t j
      have value:=Producer.large_readback c.producer scale a hau
        (Allocation.initialized_attention c.producer s) (Allocation.initialized_scratch c.producer s) h t j
      have same : InterGrid.inputs a=InterGrid.inputs s := Allocation.initialized_inputs c.producer s
      rw [same] at value
      exact value
    · intro r off hn
      have ha : ¬Allocation.attention c.producer (r,off) := fun h => hn (Or.inl h)
      have hs : ¬Allocation.scratch (Producer.splitShape c.producer) (r,off) := fun h => hn (Or.inr h)
      exact (Allocation.zeroRange_frame s "attention" (Host.attentionSize c) r off ha).trans
        ((Allocation.zeroRange_frame _ "scratch" (Host.scratchSize c) r off hs).trans
          (Allocation.large_frame c.producer scale a hau r off hn))

/-- A standalone consumer retains the caller-supplied attention and history.
Its formula has no premise that attention came from the producer. -/
theorem consumer (c : Host.Dimensions) (scale : ℝ) (s : BlockState)
    (hu : s.undef=fun _ _ => 0) :
    ∃ result, Nonempty (Kernel.GridLaunchedOrdinary (OutputGrid.kernel c.consumer scale)
      (OutputGrid.grid c.consumer) s result) ∧
      (∀ h t v, OutputGrid.returned c.consumer result h t v=
        OutputGrid.formula c.consumer (LocalReal.outputInputs OutputGrid.regions s) scale h t v) ∧
      BlockState.WriteWithin (OutputGrid.footprint c.consumer) s result :=
  OutputGrid.executes c.consumer scale s hu

#axiomsClean producer
#axiomsClean consumer
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.RealWhole
