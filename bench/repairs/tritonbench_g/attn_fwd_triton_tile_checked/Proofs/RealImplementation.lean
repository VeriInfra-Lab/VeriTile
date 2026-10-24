import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Proofs.RealBindings

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.RealInterface
open VeriTile Triton AttnFwdTriton Verified
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

theorem implements (A : FlatAlloc) (p : Parameters) (stage : p.stage=3)
    (nonoverlap : OutputLayout.NonOverlapping p) :
    (programIO A p).Implements (formula p) := by
  intro values _ s bound
  obtain ⟨t,launched,outputs,outside⟩ := GridLaunch.launch A s p (HostShape.blockD p.headDim)
    (HostShape.blockD_covers _) stage Bool.false nonoverlap
  have head : (AlgorithmStage.ordinary (selectedSource A p) (GridCoordinates.grid p)).Runs s t := by
    simpa only [AlgorithmStage.Runs,selectedSource,source_projection] using launched
  let trace : AlgorithmExecution [(AlgorithmStage.ordinary (selectedSource A p)
      (GridCoordinates.grid p))] s t := .cons head (.nil t)
  refine ⟨t,trace,?_,?_,frame A p values s t outside⟩
  · intro bounds backed
    change (∀ idx : GridIndex (GridCoordinates.grid p),
      Kernel.TraceSafe bounds (source A p).toAlgKernel (s.withGridIndex idx)) ∧ True
    refine ⟨?_,trivial⟩
    intro idx
    rw [source_projection]
    exact GridLaunch.safe A bounds s p (HostShape.blockD p.headDim) Bool.false
      (windows A p values s bounds backed idx) idx
  · intro o j _
    let c := (coordinates p).symm j
    change t.readMem A.flat (OutputLayout.address p c+A.base p.Out)=
      RealProofs.expected p.heads p.context p.headDim c.1.val c.2.1.val c.2.2.1.val c.2.2.2.val
        (logicalInputs p values)
    rw [outputs c]
    exact LogicalAgreement.expected p _ _ (agreement A p values s bound)
      _ _ _ _ c.1.isLt c.2.1.isLt c.2.2.1.isLt c.2.2.2.isLt

#axiomsClean implements
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.RealInterface
