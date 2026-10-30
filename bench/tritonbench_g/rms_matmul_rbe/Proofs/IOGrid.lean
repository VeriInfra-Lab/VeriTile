import bench.tritonbench_g.rms_matmul_rbe.Proofs.IOJob

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOGrid
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1500000
variable {α : Type} [Inhabited α]
variable (a : Arithmetic α) (zero : ZeroLaws a) (native fp8 : Bool)
variable (p : Fin 3 → Arguments) (epsilon theta : α) (order : List (Nat → Nat))
variable (vals : ArrayValues α (IO.inputs fp8 p)) (domain : IO.InputDomain fp8 p vals)
variable (even : ∀ phase,(p phase).N%2=0)
variable (apart : ∀ writer reader,(p reader).X≠(p writer).OUT ∧
  (p reader).W≠(p writer).OUT ∧ (p reader).RMS≠(p writer).OUT)
include zero domain even apart

theorem indices (launches : List (Fin order.length)) (s : State α)
    (hin : (IO.qkvIO a native fp8 p epsilon theta order).InputBinding vals s) :
    ∃ final,runSchedule (launches.map fun i=>IO.job a native fp8 p epsilon theta order[i]) s=some final ∧
      final.mem=((launches.flatMap IOJob.writers).foldl (IOEvents.put a native fp8 p epsilon theta vals) s).mem ∧
      (IO.qkvIO a native fp8 p epsilon theta order).InputBinding vals final := by
  induction launches generalizing s with
  | nil=>exact ⟨s,rfl,rfl,hin⟩
  | cons launch rest ih=>
    obtain ⟨middle,firstRun,firstMem,firstBinding⟩:=
      IOJob.success a zero native fp8 p epsilon theta order vals domain even apart s hin launch
    obtain ⟨final,restRun,restMem,restBinding⟩:=ih middle firstBinding
    refine ⟨final,?_,?_,restBinding⟩
    · simpa only [List.map_cons,runSchedule,firstRun,Option.bind_some] using restRun
    · rw [restMem,List.flatMap_cons,List.foldl_append]
      funext r off
      rw [IOEvents.scatter_at,IOEvents.scatter_at,firstMem]

theorem complete (s : State α)
    (hin : (IO.qkvIO a native fp8 p epsilon theta order).InputBinding vals s) :
    ∃ final,runSchedule ((IO.qkvIO a native fp8 p epsilon theta order).schedule vals) s=some final ∧
      final.mem=((IO.events order).foldl (IOEvents.put a native fp8 p epsilon theta vals) s).mem ∧
      (IO.qkvIO a native fp8 p epsilon theta order).InputBinding vals final := by
  have launchEq :
      (List.finRange order.length).map (fun i=>IO.job a native fp8 p epsilon theta order[i])=
        order.map (IO.job a native fp8 p epsilon theta) := by
    simpa only [List.map_map,Function.comp_def] using
      congrArg (List.map (IO.job a native fp8 p epsilon theta)) (List.map_getElem_finRange order)
  simpa only [launchEq,IO.qkvIO,IO.events,IOJob.writers] using
    indices a zero native fp8 p epsilon theta order vals domain even apart (List.finRange order.length) s hin

#axiomsClean complete
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOGrid
