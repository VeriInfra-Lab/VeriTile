import bench.repairs.tritonbench_g.parallel_attention.Proofs.IORawNative

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.IOData
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section

private theorem cell_identity {α : Type} (c : Cell α) : mapCell id c=c := by
  cases c with | mk dtype value => cases dtype <;> rfl

/-- Every original raw input, including non-Real tags, reaches the same typed
observations in the shared symbolic schedule and in the old phase-by-phase
entry/source/finish runner. Exterior raw cells agree under every phase observer. -/
theorem forward_adapter (a : Scalars NativeStage.Word) (d : NativePlans.Descriptor) (scale : ℝ)
    (s : State NativeStage.Word) :
    ∃ graph native,
      runSchedule (nativeJobs .forward d scale) (NativeStage.entry s)=some graph ∧
      runPrograms (fun p before=>NativeStage.forwardRun a d scale (withProgram p before))
        (programs d.shape 128) s=some native ∧
      (∀ o j,outputActive .forward d.shape o j →
        mapCell (nativeObserver a d o j) (graph.mem (outputPorts d.shape)[o].buf (outputWindow d.shape o j))=
          native.mem (outputPorts d.shape)[o].buf (outputWindow d.shape o j)) ∧
      (∀ R off,(dataIO NativeStage.Expr .forward d.shape (nativeJobs .forward d scale)).Outside
        (rawInputs d.shape s) R off → ∀ phase tile,
        mapCell (NativeStage.evaluate a phase d tile) (graph.mem R off)=native.mem R off) := by
  obtain ⟨graph,run,outputs,frame⟩:=native_forward_result a d scale (rawInputs d.shape s)
    (NativeStage.entry s) (raw_binding .forward d scale s)
  obtain ⟨native,oldRun,oldOutputs,oldFrame⟩:=raw_forward_result a d scale s
  refine ⟨graph,native,run,oldRun,?_,?_⟩
  · intro o j active
    have read:=outputs o j active
    change mapCell (nativeObserver a d o j)
      (mapCell id (graph.mem (outputPorts d.shape)[o].buf (outputWindow d.shape o j)))=_ at read
    rw [cell_identity] at read
    exact read.trans (oldOutputs o j active).symm
  · intro R off outside phase tile
    rw [frame R off outside,oldFrame R off outside]
    exact NativeStage.entry_finish_cell a phase d tile s R off

/-- Backward retains the original DQ/finish then DKV/finish order. The shared
schedule carries symbolic outputs, which later phases never read; all five
input regions and every exterior raw tagged cell retain their original meaning. -/
theorem backward_adapter (a : Scalars NativeStage.Word) (d : NativePlans.Descriptor) (scale : ℝ)
    (s : State NativeStage.Word) :
    ∃ graph native,
      runSchedule (nativeJobs .backward d scale) (NativeStage.entry s)=some graph ∧
      runPrograms (NativeStage.combinedRun a d scale) (programs d.shape 64) s=some native ∧
      (∀ o j,outputActive .backward d.shape o j →
        mapCell (nativeObserver a d o j) (graph.mem (outputPorts d.shape)[o].buf (outputWindow d.shape o j))=
          native.mem (outputPorts d.shape)[o].buf (outputWindow d.shape o j)) ∧
      (∀ R off,(dataIO NativeStage.Expr .backward d.shape (nativeJobs .backward d scale)).Outside
        (rawInputs d.shape s) R off → ∀ phase tile,
        mapCell (NativeStage.evaluate a phase d tile) (graph.mem R off)=native.mem R off) := by
  obtain ⟨graph,run,outputs,frame⟩:=native_backward_result a d scale (rawInputs d.shape s)
    (NativeStage.entry s) (raw_binding .backward d scale s)
  obtain ⟨native,oldRun,oldOutputs,oldFrame⟩:=raw_backward_result a d scale s
  refine ⟨graph,native,run,oldRun,?_,?_⟩
  · intro o j active
    have read:=outputs o j active
    change mapCell (nativeObserver a d o j)
      (mapCell id (graph.mem (outputPorts d.shape)[o].buf (outputWindow d.shape o j)))=_ at read
    rw [cell_identity] at read
    exact read.trans (oldOutputs o j active).symm
  · intro R off outside phase tile
    rw [frame R off outside,oldFrame R off outside]
    exact NativeStage.entry_finish_cell a phase d tile s R off

#axiomsClean forward_adapter
#axiomsClean backward_adapter
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.IOData
