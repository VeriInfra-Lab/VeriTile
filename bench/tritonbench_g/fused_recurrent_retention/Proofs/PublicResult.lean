import bench.tritonbench_g.fused_recurrent_retention.Proofs.IOSchedule
import bench.tritonbench_g.fused_recurrent_retention.Proofs.IOOutputGeometry

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.PublicData
open VeriTile.Triton FP.Structural Quoted
noncomputable section
open Classical
set_option maxHeartbeats 2000000
set_option maxRecDepth 18000
variable {α : Type} [Inhabited α]

theorem complete (mode : Mode) (a : Arithmetic α) (s : Shape) (shape : s.Admitted)
    (p : PlanBinding s) (incoming finalGradient : Fin 4 → Nat) (seed final : Bool) :
    (io mode a s p incoming finalGradient seed final).Observes
      (fun _ _ n => n.value) (expected a s p seed final) := by
  intro xs _ bounds hb initial hin
  refine ⟨schedule_safe mode a s shape p incoming finalGradient seed final xs bounds hb
    (gridPrograms s) (fun pid hp => (mem_gridPrograms s pid).mp hp) initial,?_⟩
  have domain : StoreDomain s (KernelInterface.buffers s incoming finalGradient) := ⟨shape,rfl,rfl⟩
  obtain ⟨result,ran,hm⟩ := grid_execution_memory mode a s
    (KernelInterface.buffers s incoming finalGradient) p initial seed final domain
  refine ⟨result,?_,?_,?_⟩
  · exact (schedule_run mode a s p incoming finalGradient seed final (gridPrograms s) initial).trans ran
  · intro o j active
    have inside := output_range mode s final o j active
    have agree := inputs_agree mode a s p incoming finalGradient seed final xs initial hin
    have valueEq := (sourceCell_partial a s p.plans _ seed final (kind o) (outputWindow s o j)).trans
      (Logical.partialValue_eq a p.plans agree (kind o) (outputWindow s o j) inside)
    fin_cases o
    · change mapCell Staged.Node.value (mapCell id (result.mem "o" j.1.val))=
        Cell.mk .real (partialValue a s p.plans (logicalInputs a s xs) seed final .output j.1.val)
      change writtenRange mode s final "o" j.1.val at inside
      change sourceCell a s p.plans _ seed final .output j.1.val=
        partialValue a s p.plans (logicalInputs a s xs) seed final .output j.1.val at valueEq
      rw [hm,if_pos inside]
      simpa only [payload,Region.mk.injEq,String.reduceEq,reduceIte,mapCell,mapValue,id_eq,Staged.atom]
        using congrArg (Cell.mk .real) valueEq
    · change mapCell Staged.Node.value (mapCell id (result.mem "dq" j.1.val))=
        Cell.mk .real (partialValue a s p.plans (logicalInputs a s xs) seed final .dq j.1.val)
      change writtenRange mode s final "dq" j.1.val at inside
      change sourceCell a s p.plans _ seed final .dq j.1.val=
        partialValue a s p.plans (logicalInputs a s xs) seed final .dq j.1.val at valueEq
      rw [hm,if_pos inside]
      simpa only [payload,Region.mk.injEq,String.reduceEq,reduceIte,mapCell,mapValue,id_eq,Staged.atom]
        using congrArg (Cell.mk .real) valueEq
    · change mapCell Staged.Node.value (mapCell id (result.mem "dk" j.1.val))=
        Cell.mk .real (partialValue a s p.plans (logicalInputs a s xs) seed final .dk j.1.val)
      change writtenRange mode s final "dk" j.1.val at inside
      change sourceCell a s p.plans _ seed final .dk j.1.val=
        partialValue a s p.plans (logicalInputs a s xs) seed final .dk j.1.val at valueEq
      rw [hm,if_pos inside]
      simpa only [payload,Region.mk.injEq,String.reduceEq,reduceIte,mapCell,mapValue,id_eq,Staged.atom]
        using congrArg (Cell.mk .real) valueEq
    · change mapCell Staged.Node.value (mapCell id (result.mem "dv" j.1.val))=
        Cell.mk .real (partialValue a s p.plans (logicalInputs a s xs) seed final .dv j.1.val)
      change writtenRange mode s final "dv" j.1.val at inside
      change sourceCell a s p.plans _ seed final .dv j.1.val=
        partialValue a s p.plans (logicalInputs a s xs) seed final .dv j.1.val at valueEq
      rw [hm,if_pos inside]
      simpa only [payload,Region.mk.injEq,String.reduceEq,reduceIte,mapCell,mapValue,id_eq,Staged.atom]
        using congrArg (Cell.mk .real) valueEq
    · change mapCell Staged.Node.value (mapCell id (result.mem "final_state" j.1.val))=
        Cell.mk .real (partialValue a s p.plans (logicalInputs a s xs) seed final .finalState j.1.val)
      change writtenRange mode s final "final_state" j.1.val at inside
      change sourceCell a s p.plans _ seed final .finalState j.1.val=
        partialValue a s p.plans (logicalInputs a s xs) seed final .finalState j.1.val at valueEq
      rw [hm,if_pos inside]
      simpa only [payload,Region.mk.injEq,String.reduceEq,reduceIte,mapCell,mapValue,id_eq,Staged.atom]
        using congrArg (Cell.mk .real) valueEq
  · intro r i outside
    rw [hm,if_neg (outside_written mode a s p incoming finalGradient seed final xs r i outside)]

#axiomsClean complete
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.PublicData
