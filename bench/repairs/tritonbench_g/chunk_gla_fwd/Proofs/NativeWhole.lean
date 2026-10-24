import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeScope

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeWhole
open VeriTile Triton FP.Structural
open FPStorage (Memory payload)
open NativeScope
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 1000000

def Outside (c : Host.Dimensions) (r : RegionName) (off : Nat) : Prop :=
  (r≠"attention" ∨ Host.attentionSize c≤off) ∧
    (c.heads*c.T=0 ∨ c.K≤256 ∨ r≠"scratch" ∨ Host.scratchSize c≤off)

theorem producer (M : Algebra α) (D : FPOutput.DotPrimitive α) (c : Host.Dimensions)
    (scale : ℝ) (s : State α) (m : Memory α)
    (inputs : ∀ r off,inputScope c r off → (s.mem r off).read .real=m r off) :
    ∃ final,runFillSchedule (NativeLaunch.producer M D c scale) s=some final ∧
      (∀ off,off<Host.attentionSize c → payload (final.mem "attention" off)=
        Host.producerReference M D c scale m "attention" off) ∧
      (∀ r off,Outside c r off → final.mem r off=s.mem r off) := by
  let initialized := (NativeLaunch.zero M "attention" (Host.attentionSize c)).apply s
  let initialMemory := Host.zeroRange M m "attention" (Host.attentionSize c)
  have initial : ∀ r off,producerScope c Bool.false r off →
      producerObserve r (initialized.mem r off)=initialMemory r off := by
    refine fill_reference (inputScope c) (producerScope c Bool.false) producerObserve M
      "attention" (Host.attentionSize c) ?_ ?_ s m ?_
    · intro r off inside
      rcases inside with input|attention|scratch
      · exact Or.inl input
      · exact Or.inr attention
      · simp at scratch
    · rfl
    · intro r off inside
      have other : r≠"attention" := by
        rcases inside.1 with rfl|rfl|rfl <;> decide
      simpa only [producerObserve,if_neg other] using inputs r off inside
  by_cases empty : c.heads*c.T=0
  · refine ⟨initialized,?_,?_,?_⟩
    · simp only [NativeLaunch.producer,empty,if_true,runFillSchedule,FillStage.run,Option.bind_some]
      rfl
    · intro off inside
      simpa only [producerObserve,if_pos rfl,Host.producerReference,if_pos empty,initialMemory]
        using initial "attention" off (Or.inr (Or.inl ⟨rfl,inside⟩))
    · intro r off outside
      exact fill_frame M "attention" (Host.attentionSize c) s r off outside.1
  · obtain ⟨inter,interRan,interValues,interFrame⟩ := NativeGrid.grid_reference
      (producerScope c Bool.false) producerObserve M D (.inter c.producer scale)
      (inter_ports (α:=α) c scale Bool.false).1
      (inter_ports (α:=α) c scale Bool.false).2.1
      (inter_ports (α:=α) c scale Bool.false).2.2 initialized initialMemory initial
    dsimp only [initialized] at interRan
    by_cases small : c.K≤256
    · obtain ⟨final,finalRan,values,frame⟩ := NativeGrid.grid_reference
        (producerScope c Bool.false) producerObserve M D (.intra c.producer c.intraBK scale)
        (intra_ports (α:=α) c scale Bool.false).1
        (intra_ports (α:=α) c scale Bool.false).2.1
        (intra_ports (α:=α) c scale Bool.false).2.2 inter
        (FPStages.reference M D (.inter c.producer scale) initialMemory) interValues
      refine ⟨final,?_,?_,?_⟩
      · simp only [NativeLaunch.producer,if_neg empty,if_pos small,runFillSchedule,FillStage.run,
          Option.bind_some,NativeGrid.run_append,interRan,finalRan]
      · intro off inside
        simpa only [producerObserve,if_pos rfl,Host.producerReference,if_neg empty,if_pos small,initialMemory]
          using values "attention" off (Or.inr (Or.inl ⟨rfl,inside⟩))
      · intro r off outside
        exact (frame r off outside.1).trans ((interFrame r off outside.1).trans
          (fill_frame M "attention" (Host.attentionSize c) s r off outside.1))
    · let scratch := (NativeLaunch.zero M "scratch" (Host.scratchSize c)).apply inter
      let scratchMemory := Host.zeroRange M (FPStages.reference M D (.inter c.producer scale) initialMemory)
        "scratch" (Host.scratchSize c)
      have scratchValues : ∀ r off,producerScope c Bool.true r off →
          producerObserve r (scratch.mem r off)=scratchMemory r off := by
        apply fill_reference (producerScope c Bool.false) (producerScope c Bool.true)
          producerObserve M "scratch" (Host.scratchSize c) _ _ inter
          (FPStages.reference M D (.inter c.producer scale) initialMemory) interValues
        · intro r off inside
          rcases inside with input|attention|written
          · exact Or.inl (Or.inl input)
          · exact Or.inl (Or.inr (Or.inl attention))
          · exact Or.inr written.2
        · simp [producerObserve,Cell.read_same]
      obtain ⟨split,splitRan,splitValues,splitFrame⟩ := NativeGrid.grid_reference
        (producerScope c Bool.true) producerObserve M D (.split c.split scale)
        (split_ports (α:=α) c scale).1 (split_ports (α:=α) c scale).2.1
        (split_ports (α:=α) c scale).2.2 scratch scratchMemory scratchValues
      dsimp only [scratch] at splitRan
      obtain ⟨final,finalRan,values,frame⟩ := NativeGrid.grid_reference
        (producerScope c Bool.true) producerObserve M D (.merge c.split)
        (merge_ports (α:=α) c).1 (merge_ports (α:=α) c).2.1 (merge_ports (α:=α) c).2.2 split
        (FPStages.reference M D (.split c.split scale) scratchMemory) splitValues
      refine ⟨final,?_,?_,?_⟩
      · simp only [NativeLaunch.producer,if_neg empty,if_neg small,runFillSchedule,FillStage.run,
          Option.bind_some,NativeGrid.run_append,interRan,splitRan,finalRan]
      · intro off inside
        simpa only [producerObserve,if_pos rfl,Host.producerReference,if_neg empty,if_neg small,
            initialMemory,scratchMemory]
          using values "attention" off (Or.inr (Or.inl ⟨rfl,inside⟩))
      · intro r off outside
        have hs : r≠"scratch" ∨ Host.scratchSize c≤off :=
          (outside.2.resolve_left empty).resolve_left small
        exact (frame r off outside.1).trans ((splitFrame r off hs).trans
          ((fill_frame M "scratch" (Host.scratchSize c) inter r off hs).trans
          ((interFrame r off outside.1).trans (fill_frame M "attention" (Host.attentionSize c) s r off outside.1))))

theorem consumer (M : Algebra α) (D : FPOutput.DotPrimitive α) (c : Host.Dimensions)
    (scale : ℝ) (s : State α) :
    ∃ final,runFillSchedule (NativeLaunch.consumer M D c scale) s=some final ∧
      (∀ r off,consumerObserve r (final.mem r off)=
        Host.consumerReference M D c scale (consumerMemory s) r off) ∧
      (∀ (r : RegionName) off,r≠"o" ∨ Host.outputSize c≤off → final.mem r off=s.mem r off) := by
  by_cases empty : Host.outputSize c=0
  · refine ⟨s,?_,?_,fun _ _ _=>rfl⟩
    · simp only [NativeLaunch.consumer,if_pos empty,runFillSchedule]
    · intro r off
      simp only [Host.consumerReference,if_pos empty,consumerMemory]
  · obtain ⟨final,ran,values,frame⟩ := NativeGrid.grid_reference (fun _ _=>True)
      consumerObserve M D (.output c.consumer scale)
      (output_ports (α:=α) c scale).1 (output_ports (α:=α) c scale).2.1
      (output_ports (α:=α) c scale).2.2 s (consumerMemory s) (fun _ _ _=>rfl)
    refine ⟨final,?_,?_,frame⟩
    · simpa only [NativeLaunch.consumer,if_neg empty] using ran
    · intro r off
      simpa only [Host.consumerReference,if_neg empty] using values r off True.intro

#axiomsClean producer
#axiomsClean consumer
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeWhole
