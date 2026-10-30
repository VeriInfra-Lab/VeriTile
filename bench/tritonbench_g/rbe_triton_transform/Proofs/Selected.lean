import bench.tritonbench_g.rbe_triton_transform.Proofs.SelectedExecution
namespace VeriTile.Bench.Repairs.RbeTritonTransform.Selected
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem complete {α : Type} [Inhabited α] (ops : Operations α) (p : Source.Parameters) (s : State α) :
    ∃ t, exec ops.backend (source p) s=some t ∧
      t.mem=(Source.written ops.backend p s).mem := by
  refine ⟨(execution ops.backend p s).val,(execution ops.backend p s).property,?_⟩
  simp +contextual [execution,Source.written,Source.halfScatter,
    VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy.masked_scatter_memory,
    Source.value,Source.component,Source.phase,Source.active,Source.inputOffset,
    Source.outputOffset,Source.row,Source.column,Source.parts,
    VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy.masked_scatter_regs,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Region.cast,
    integralFloorDiv,integralMod,and_assoc,and_left_comm,and_comm,Operations.backend]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RbeTritonTransform.Selected
